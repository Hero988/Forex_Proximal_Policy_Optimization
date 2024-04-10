import pandas as pd
import datetime
import MetaTrader5 as mt5
from datetime import datetime
import gymnasium as gym
import pytz
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from gym import spaces
import pickle
import torch
import pandas_ta as ta
import torch.optim as optim
import os
from torch.distributions import Categorical
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define the ForexTradingEnv class, inheriting from gym.Env to create a custom trading environment
class ForexTradingEnv(gym.Env):
    # Metadata for the environment, specifying available render modes
    metadata = {'render.modes': ['human']}

    # Constructor with parameters for initializing the environment
    def __init__(self, data, initial_balance=10000, leverage=100, transaction_cost=0.0002, stop_loss_percent=0.5, take_profit_percent=1, lot_size=1000):
        # Call the constructor of the superclass
        super(ForexTradingEnv, self).__init__()

        # Initialize environment variables with provided arguments or defaults
        self.data = data  # Data for trading, typically historical price data
        self.initial_balance = initial_balance  # Starting balance for the trading account
        self.current_balance = initial_balance  # Current balance, initialized to the starting balance
        self.max_daily_loss_limit = initial_balance * 0.05  # Set the maximum daily loss limit to 5% of the initial balance
        self.max_loss_limit = initial_balance * 0.1  # Set the overall loss limit to 10% of the initial balance
        self.profit_target = initial_balance * 0.1  # Set a profit target of 10% of the initial balance
        self.balance = initial_balance  # Current balance, repeated initialization for clarity
        self.peak_balance = initial_balance  # Peak balance achieved, initialized to the starting balance
        self.equity = initial_balance  # Equity, initially equal to the starting balance
        self.leverage = leverage  # Leverage available for trading
        self.transaction_cost = transaction_cost  # Cost per transaction
        self.stop_loss_percent = stop_loss_percent  # Stop loss threshold in percentage
        self.take_profit_percent = take_profit_percent  # Take profit threshold in percentage
        self.lot_size = lot_size  # Size of a single trade lot
        self.position = 'neutral'  # Current trading position, initialized as neutral (no open positions)
        self.is_open_position = False  # Flag to indicate if a trading position is open
        self.accumulated_lot_size = 0  # Total size of the open position
        self.entry_price = None  # Price at which the current position was opened
        self.position_open_step = None  # Step number when the current position was opened
        self.sl_price = 0  # Stop loss price
        self.tp_price = 0  # Take profit price
        self.current_step = 0  # Current step in the environment
        self.trade_history = []  # History of trades made
        self.pnl_history = [0]  # History of profit and loss, starting with 0
        self.trade_profit = 0  # Profit from the current trade, initialized to 0
        self.position_closed_recently = False  # Flag to indicate if a position was closed recently
        self.max_position_duration = 288  # Max duration for which a position can stay open
        self.window_size = 288  # Size of the window for observation data
        self.num_features = 25  # Number of features in the observation data

        # Define the action space of the environment
        self.action_space = spaces.Discrete(4)  # 0 - Buy, 1 - Sell, 2 - Close, 3 - Hold
        
        # Define the observation space of the environment
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size * self.num_features,), dtype=np.float32)  # Observation space defined as a box with values ranging from 0 to infinity, shaped according to the window size and number of features

    # Method to save the current state of the environment to a file
    def save_state(self, filename):
        with open(filename, 'wb') as f:  # Open the file in write-binary mode
            pickle.dump(self.__dict__, f)  # Dump the environment's dictionary to the file

    # Method to load the state of the environment from a file
    def load_state(self, filename):
        with open(filename, 'rb') as f:  # Open the file in read-binary mode
            state_dict = pickle.load(f)  # Load the dictionary from the file
            self.__dict__.update(state_dict)  # Update the environment's dictionary with the loaded state

    # Method to update the current profit or loss based on the current price
    def update_current_pnl(self, current_price):
        if self.position == 'long':  # If the current position is long
            # Calculate P&L for long position
            self.current_pnl = (current_price - self.entry_price) * self.accumulated_lot_size - self.transaction_cost
        elif self.position == 'short':  # If the current position is short
            # Calculate P&L for short position
            self.current_pnl = (self.entry_price - current_price) * self.accumulated_lot_size - self.transaction_cost
        else:  # If there is no open position
            self.current_pnl = 0  # Set current P&L to 0

    # Define a method to execute a trade based on the action, current market price, and trade confidence
    def execute_trade(self, action, current_price, confidence):
        trade_executed = False  # Initialize a flag to track if a trade was executed
        
        # Define the range for lot sizes based on the confidence level of the trade
        min_lot_size = 1000  # Minimum lot size for a trade
        max_lot_size = 10000  # Maximum lot size for a trade
        
        # Calculate the lot size for the current trade based on the confidence level, scaling it between min_lot_size and max_lot_size
        self.lot_size = (confidence * (max_lot_size - min_lot_size)) + min_lot_size

        # Update the current profit and loss (P&L) based on the current market price
        self.update_current_pnl(current_price)
        
        # Append the current P&L to the history of P&Ls to keep a record over time
        self.pnl_history.append(self.current_pnl)

        # If the current balance is greater than the peak balance, update the peak balance to the current balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Calculate the accumulated lot size by summing the lot sizes of all open trades from the trade history
        self.accumulated_lot_size = sum([trade.get('lot_size') for trade in self.trade_history if trade.get('is_open', True)])

        # Determine if there is an open position by checking if any trade in the history is marked as open
        self.is_open_position = any(trade.get('is_open', False) for trade in self.trade_history)

        # If the action is 2 (close), call the method to close any open position at the current market price
        if action == 2:
            self.close_position_any(current_price)

        # If the action is 3 (hold), log the decision to hold without making a trade by appending a record to the trade history
        if action == 3:
            self.trade_history.append({
                'Step': self.current_step,  # Record the current step (time) of the hold action
                'Action': 'Hold',  # Indicate the action taken was to hold
                'Position': self.position,  # Record the current trading position
                'Price': current_price,  # Record the current market price at the time of holding
                'Balance': self.balance,  # Record the current balance of the account
                'Date': self.data.index[self.current_step].strftime('%Y-%m-%d'),  # Format and record the current date
                'lot_size': 0,  # Indicate that no lot size is applicable for hold actions
            })
         
        # Check if there is an open position
        if self.is_open_position:
            # Handle closing long or short positions based on the action
            if self.position == 'long' and action == 1:  # If intending to sell and close the long position
                trade_executed = True  # Mark that a trade was executed
                self.close_position_any(current_price)  # Close the current long position at the current price
                self.open_position(current_price, 'short', self.lot_size)  # Open a new short position with the current lot size
            elif self.position == 'short' and action == 0:  # If intending to buy and close the short position
                trade_executed = True  # Mark that a trade was executed
                self.close_position_any(current_price)  # Close the current short position at the current price
                self.open_position(current_price, 'long', self.lot_size)  # Open a new long position with the current lot size
            elif self.position == 'long' and action == 0:
                trade_executed = True  # Mark that a trade was executed
                self.open_position(current_price, 'long', self.lot_size)  # Open a new long position with the current lot size
            elif self.position == 'short' and action == 1:
                trade_executed = True  # Mark that a trade was executed
                self.open_position(current_price, 'short', self.lot_size)  # Open a new short position with the current lot size

        # Check if the accumulated lot size exceeds the maximum allowed limit
        if self.accumulated_lot_size >= 100000:
            self.close_position_any(current_price)  # Close any position if the accumulated lot size is too large
            # Allow closing and opening of new positions based on the action and current position
            if self.position != 'neutral':  # Ensure there's an active position
                if self.position == 'long' and action == 1:  # If the action is to sell from a long position
                    trade_executed = True
                    self.close_position_any(current_price)  # Close the long position
                    self.open_position(current_price, 'short', self.lot_size)  # Open a new short position
                elif self.position == 'short' and action == 0:  # If the action is to buy from a short position
                    trade_executed = True
                    self.close_position_any(current_price)  # Close the short position
                    self.open_position(current_price, 'long', self.lot_size)  # Open a new long position

        # Assuming no early return, we proceed to check if new positions can be opened
        # Given no open positions, we can open a new one based on the action
        if not self.is_open_position or self.position == 'neutral':
            if action == 0:  # If the action is to buy
                trade_executed = True  # Mark that a trade was executed
                self.open_position(current_price, 'long', self.lot_size)  # Open a new long position
            elif action == 1:  # If the action is to sell
                trade_executed = True  # Mark that a trade was executed
                self.open_position(current_price, 'short', self.lot_size)  # Open a new short position

        return trade_executed  # Return whether a trade was executed or not

    # Define a method to open a new trading position
    def open_position(self, current_price, position_type, lot_size):
        """Opens a new trading position."""
        
        # Calculate the cost of opening the position based on transaction cost and lot size
        cost = self.transaction_cost * lot_size
        
        # Deduct the opening cost from the current balance
        self.current_balance -= cost
        
        # Calculate and set the margin required for the position
        self.margin = (lot_size / current_price) / self.leverage
        
        # Set the type of the position (long or short)
        self.position = position_type
        
        # Record the price at which the position is opened
        self.entry_price = current_price
        
        # Update the total accumulated lot size with the lot size of the new position
        self.accumulated_lot_size += lot_size
        
        # Indicate that there is now an open position
        self.is_open_position = True
        
        # Record the current step as the step at which the position was opened
        self.position_open_step = self.current_step

        # If the position is long, calculate and set the stop loss and take profit prices accordingly
        if position_type == 'long':
            self.sl_price = current_price * (1 - self.stop_loss_percent / 100)  # Calculate stop loss price for a long position
            self.tp_price = current_price * (1 + self.take_profit_percent / 100)  # Calculate take profit price for a long position
        
        # If the position is short, calculate and set the stop loss and take profit prices accordingly
        elif position_type == 'short':
            self.sl_price = current_price * (1 + self.stop_loss_percent / 100)  # Calculate stop loss price for a short position
            self.tp_price = current_price * (1 - self.take_profit_percent / 100)  # Calculate take profit price for a short position

        # Deduct the cost again from the balance for some reason (seems like an error or redundant step)
        self.balance -= cost

        # Append a new entry to the trade history with details of the open position
        self.trade_history.append({
            'Action': 'Open ' + position_type.capitalize(),  # Record the action taken (Open Long/Open Short)
            'Position': self.position,  # Record the type of the position opened
            'Price': current_price,  # Record the opening price of the position
            'Balance': self.balance,  # Record the balance after opening the position
            'lot_size': lot_size,  # Record the lot size of the position
            'Date': datetime.now().strftime('%Y-%m-%d'),  # Record the current date of the position opening
            'SL Price': self.sl_price,  # Record the calculated stop loss price
            'TP Price': self.tp_price,  # Record the calculated take profit price
            'is_open': True,  # Indicate that this trade entry is currently open
        })

    # Define a method to close any open trading position at a given price
    def close_position_any(self, current_price):
        # Check if there is an open position to close
        if self.position in ['long', 'short']:
            # Check if the position is a long position
            if self.position == 'long':
                self.position_closed_recently = True  # Mark that a position was recently closed
                # Calculate the profit or loss for the long position
                self.trade_profit = (current_price - self.entry_price) * self.lot_size - self.transaction_cost
            # Check if the position is a short position
            elif self.position == 'short':
                self.position_closed_recently = True  # Mark that a position was recently closed
                # Calculate the profit or loss for the short position
                self.trade_profit = (self.entry_price - current_price) * self.lot_size - self.transaction_cost
            else:
                # This else block seems redundant due to the initial if condition and can be considered unreachable
                self.position_closed_recently = True  # Ensure the flag is set even if neither long nor short (redundant)
                self.trade_profit = 0  # Set profit to 0 as a fallback

            # Iterate through the trade history to find and close the open position
            for trade in self.trade_history:
                # Check if the trade is marked as open
                if trade.get('is_open', False):  # Default to False if 'is_open' key is not found
                    trade['is_open'] = False  # Mark the trade as closed

            self.accumulated_lot_size = 0  # Reset the accumulated lot size to 0 as the position is closed

            # Update the account balance with the profit or loss from closing the position
            self.balance += self.trade_profit

            # Record the closing of the position in the trade history
            self.trade_history.append({
                'Step': self.current_step,  # Record the step at which the position was closed
                'Action': 'Closed Position',  # Indicate the action taken was closing a position
                'Position': self.position,  # Record the type of the position that was closed
                'Price': current_price,  # Record the price at which the position was closed
                'Profit': self.trade_profit,  # Record the profit or loss from the trade
                'Balance': self.balance,  # Record the updated balance after closing the position
                'Date': self.data.index[self.current_step].strftime('%Y-%m-%d'),  # Record the date of the position closing
                'lot_size': self.accumulated_lot_size,  # Record the lot size of the position that was closed
                'Closed Balance': self.balance  # Record the balance after the position was closed
            })

            # Reset attributes related to the position
            self.position = 'neutral'  # Reset the position status to neutral
            self.entry_price = None  # Clear the entry price
            self.sl_price = 0  # Reset the stop loss price
            self.tp_price = None  # Clear the take profit price
            self.margin = 0  # Reset the margin used
            self.position_open_step = None  # Clear the step at which the position was opened

            # Update and record the current profit and loss, which should be 0 as the position is now closed
            self.update_current_pnl(current_price)  # This call may not effectively change pnl since the position is closed
            self.pnl_history.append(self.current_pnl)  # Append the final P&L for the closed position to the history
            # Optional print statement for logging the close action (commented out)
            #print(f"Close Position Manually at: {self.data.index[self.current_step].strftime('%Y-%m-%d')} with price: {current_price}")
        else:
            # If there's no open position to close, record this attempt in the trade history
            self.trade_history.append({
                'Action': 'No open position to close',  # Indicate that there was an attempt to close a non-existent position
                'Position': self.position,  # Record the current position status, which should be neutral
                'Price': current_price,  # Record the current price at which the close attempt was made
                'Balance': self.balance,  # Record the current balance
                'lot_size': 0,  # Indicate that no lot size is involved in this action
                'Date': datetime.now().strftime('%Y-%m-%d'),  # Record the date of the attempt
                'SL Price': self.sl_price,  # Record the current stop loss price
                'TP Price': self.tp_price,  # Record the current take profit price
                'is_open': True,  # Indicates that the trade is open
            })

    # Define a method to reset attributes related to any open trading position
    def reset_position_attributes(self):
        # Reset the position to 'neutral', indicating no open positions
        self.position = 'neutral'
        # Clear the entry price as there's no open position
        self.entry_price = None
        # Reset the stop loss and take profit prices to their default values
        self.sl_price = 0
        self.tp_price = 0
        # Reset the accumulated lot size to zero
        self.accumulated_lot_size = 0
        # Reset the margin used for the position to zero
        self.margin = 0
        # Clear the step at which the position was opened
        self.position_open_step = None


    # Define a method to reset the environment to its initial state
    def reset(self):
        # Reset the account balance to the initial balance
        self.balance = self.initial_balance
        # Reset the equity to the initial balance
        self.equity = self.initial_balance
        # Reset the current step to zero
        self.current_step = 0
        # Reset the position status to 'neutral'
        self.position = 'neutral'
        # Clear the entry price
        self.entry_price = None
        # Reset the trade history to an empty list
        self.trade_history = []
        # Reset the profit and loss history, starting with an initial value of 0
        self.pnl_history = [0]
        # Generate the next observation based on the reset state
        observation = self._next_observation()
        
        assert observation.shape == (self.observation_space.shape[0],), f"Observation shape mismatch: {observation.shape} != {self.observation_space.shape}"
        # Return the observation to the caller, typically for starting a new episode
        return observation


    # Define a method to get the next observation from the data
    def _next_observation(self):
        # Initialize the count of missing rows to 0
        missing_rows = 0
        
        # Calculate the end index for the observation window, ensuring it doesn't exceed the length of the data
        end_idx = min(self.current_step + self.window_size, len(self.data))
        
        # Slice the DataFrame from the current step up to the calculated end index
        obs_df = self.data.iloc[self.current_step:end_idx]
        
        # Check if the obtained slice is smaller than the window size
        if len(obs_df) < self.window_size:
            # Calculate how many rows are missing to reach the window size
            missing_rows = self.window_size - len(obs_df)
            # Create a DataFrame of zeros to pad the observation to the correct size
            padding = pd.DataFrame(np.zeros((missing_rows, self.data.shape[1])), columns=self.data.columns)
            # Concatenate the padding DataFrame to the end of the observation DataFrame
            obs_df = pd.concat([obs_df, padding], ignore_index=True)
        
        # Flatten the DataFrame into a 1D array for use in the environment
        flattened_obs = obs_df.values.flatten()
        
        # Increment the current step
        self.current_step += 1

        # Return the flattened observation array
        return flattened_obs

    # Define a method to take a step in the environment based on an action
    def step(self, action, trade_history, confidence):
        # Reset the flag indicating whether a position was closed recently
        self.position_closed_recently = False  

        # Determine if the simulation is done based on the remaining data
        done = self.current_step >= len(self.data) - 5

        # Get the current price from the data using the 'close' column
        current_price = self.data['close'].iloc[self.current_step]
        # Format the current date as a string
        current_date = self.data.index[self.current_step].strftime('%Y-%m-%d')
        # Execute the trade based on the provided action and confidence
        self.execute_trade(action, current_price, confidence)     

        # Close the position if the price crosses the stop loss or take profit thresholds
        if self.position == 'long':
            if current_price <= self.sl_price or current_price >= self.tp_price:
                self.close_position_any(current_price)
        elif self.position == 'short':
            if current_price >= self.sl_price or current_price <= self.tp_price:
                self.close_position_any(current_price)

        # Calculate the reward for the current step
        step_reward = self.calculate_step_reward(action, trade_history, self.current_step)

        # Log the trade action taken
        self.log_trade(self.current_step, action, current_price, self.balance, current_date, self.lot_size)

        # Get the next observation after taking the step
        next_observation = self._next_observation()
        
        # Ensure the shape of the next observation matches the expected observation space shape
        assert next_observation.shape == (self.observation_space.shape[0],), f"Next observation shape mismatch: {next_observation.shape} != {self.observation_space.shape}"
        # Return the next observation, the step reward, the done flag, and an empty info dict
        return next_observation, step_reward, done, {}

    # Define a method to evaluate and summarize the trading performance
    def evaluate_performance(self):
        # Initialize peak balance, drawdown, and highest daily loss variables
        peak = self.initial_balance  # Set the initial peak to the initial balance
        drawdown = 0  # Initial drawdown value
        highest_daily_loss = 0  # Initial highest daily loss value
        
        # Create a list to hold daily balances, starting with the initial balance
        daily_balances = [self.initial_balance]

        # Iterate through each trade in the trade history
        for trade in self.trade_history:
            current_balance = trade['Balance']  # Extract the balance after the trade
            daily_balances.append(current_balance)  # Append the balance to the daily balances list
            
            # Update the peak balance if the current balance exceeds the peak
            if current_balance > peak:
                peak = current_balance
            
            # Calculate the drawdown as a percentage
            drawdown = max(drawdown, (peak - current_balance) / peak * 100)

            # Calculate daily loss if there are at least two daily balances to compare
            if len(daily_balances) > 1:
                daily_loss = (daily_balances[-2] - daily_balances[-1]) / daily_balances[-2] * 100
                highest_daily_loss = min(highest_daily_loss, daily_loss)  # Update highest daily loss if this loss is greater

        # Convert the highest daily loss into a positive value for reporting
        magnitude_of_highest_daily_loss = abs(highest_daily_loss)
        # Round the magnitude of the highest daily loss to two decimal places
        magnitude_of_highest_daily_loss = round(magnitude_of_highest_daily_loss, 2)
        
        # Initialize variables to calculate additional metrics
        total_profit = 0  # Sum of all profitable trades
        total_loss = 0  # Sum of all losing trades
        win_count = 0  # Count of profitable trades
        loss_count = 0  # Count of losing trades

        # Iterate through each trade to calculate profits, losses, and counts
        for trade in self.trade_history:
            if 'Profit' in trade:
                if trade['Profit'] > 0:
                    total_profit += trade['Profit']  # Add to total profit
                    win_count += 1  # Increment win count
                else:
                    total_loss += abs(trade['Profit'])  # Add to total loss
                    loss_count += 1  # Increment loss count

        # Calculate the profit factor, win rate, and other metrics
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')  # Profit factor calculation
        number_of_closed_trades = win_count + loss_count  # Total number of trades that were closed
        percent_profitable = (win_count / number_of_closed_trades * 100) if number_of_closed_trades > 0 else 0  # Calculate the percentage of profitable trades

        # Calculate final profit by subtracting the initial balance from the final balance
        final_profit = self.balance - self.initial_balance

        # Return a dictionary summarizing the performance metrics
        return {
                "Final Balance": self.balance,  # The final account balance
                "Final Profit": final_profit,  # The net profit or loss
                "Maximum Drawdown": f"{drawdown:.2f}%",  # The maximum drawdown in percentage
                "Highest Daily Loss": f"{highest_daily_loss:.2f}%",  # The highest daily loss in percentage
                "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "Inf",  # The profit factor
                "Number of Closed Trades": number_of_closed_trades,  # The total number of trades closed
                "Percent Profitable": f"{percent_profitable:.2f}%",  # The percentage of trades that were profitable
                "Magnitude of loss": magnitude_of_highest_daily_loss  # The magnitude of the highest daily loss
            }

    # Define a method to log the details of a trade
    def log_trade(self, step, action, price, balance, date, lot_size):
        # Check if the action is represented as an integer (e.g., 0 for Buy, 1 for Sell)
        if isinstance(action, int):
            # Convert the integer action to a human-readable string
            action_str = {0: 'Buy', 1: 'Sell', 2: 'Close'}.get(action, 'Unknown')
        else:
            # If the action is already a string, use it directly
            action_str = action
        # Optional print statement to log the trade details (commented out)
        #print(f"Step: {step}, Action: {action_str}, Price: {price}, Balance: {balance}, Date: {date}, Current Lot Size: {lot_size}")
            
    def calculate_step_reward(self, action, trade_history, step_number):
        if not self.trade_history or len(self.trade_history) < 2:
            # Penalize for lack of activity or insufficient history to assess performance
            return -5  # Small negative reward to encourage activity

        # Calculate the difference in balance between the last two trades
        change_in_balance = self.trade_history[-1]['Balance'] - self.trade_history[-2]['Balance']

        # Base reward for executing a trade, could be adjusted based on action effectiveness
        base_reward = 1

        # Calculate profit or loss percentage relative to the account balance
        profit_or_loss_percent = (change_in_balance / self.trade_history[-2]['Balance']) * 100

        if change_in_balance > 0:
            # Scale the reward positively by the profit percentage
            # Adding base reward to ensure a positive outcome from profitable actions
            reward = base_reward + (profit_or_loss_percent * 0.1)  # Scaling factor for reward based on profit percentage
        elif change_in_balance == 0:
            # Small negative reward for no change in balance to encourage profitable actions
            reward = -1
        else:
            # Larger penalty for losses, scaled by the loss percentage
            # The penalty is larger in magnitude compared to the reward for profit
            reward = -2 * abs(profit_or_loss_percent * 0.1)  # Scaling factor for penalty based on loss percentage

        # Additional considerations
        if self.position_closed_recently and self.trade_profit < 0:
            # Additional penalty if the position was closed recently with a loss
            reward -= 10  # This could be adjusted based on the severity of the loss

        # Adjust reward for risk management
        if self.position == 'neutral' and action == 3:  # Assuming action 3 is 'Hold'
            # Small penalty for holding in a neutral position; might indicate missed opportunities
            reward -= 2

        return reward

# Define the ActorCritic class which inherits from nn.Module, the base class for all neural network modules in PyTorch
class ActorCritic(nn.Module):
    # The constructor takes input dimensions and action dimensions as arguments
    def __init__(self, input_dim, action_dim):
        # Call the constructor of the parent class (nn.Module)
        super(ActorCritic, self).__init__()
        
        # Define shared layers that will be used to extract features from the input state
        # These layers are common to both the actor and critic components
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),  # First linear layer that transforms the input state to a 64-dimensional space
            nn.ReLU(),  # Apply ReLU activation function to introduce non-linearity
            nn.Linear(64, 32),  # Second linear layer, mapping the 64-dimensional space to another 64-dimensional space
            nn.ReLU()  # Another ReLU activation function
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic head
        self.critic = nn.Linear(32, 1)

    def forward(self, state):
        shared_out = self.shared_layers(state)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value

# Define a class PPOAgent to encapsulate the behavior and data of a Proximal Policy Optimization agent
class PPOAgent:
    # Constructor with default parameters for the agent
    def __init__(self, input_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, clip_epsilon=0.2, gae_lambda=0.95, mode='train'):
        # Initialize the ActorCritic network with specified input and action dimensions
        self.model = ActorCritic(input_dim, action_dim)
        # Use Adam optimizer for updating the parameters of the model, with learning rate for the actor
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_actor)
        # Discount factor for future rewards
        self.gamma = gamma
        # Epsilon value for clipping in the PPO loss
        self.clip_epsilon = clip_epsilon
        # Lambda for Generalized Advantage Estimation
        self.gae_lambda = gae_lambda
        # Store input and action dimensions
        self.input_dim = input_dim
        self.action_dim = action_dim
        # Store learning rates for actor and critic
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

    def set_train_mode(self):
        """Sets the agent's mode to training."""
        self.mode = 'train'
        self.model.train()  # Set the model to training mode
    
    def set_eval_mode(self):
        """Sets the agent's mode to evaluation."""
        self.mode = 'eval'
        self.model.eval()  # Set the model to evaluation mode

    def select_action(self, state):
        """Selects an action based on the current state and the agent's mode."""
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to PyTorch tensor and add batch dimension
        
        with torch.no_grad():  # Disable gradient calculation for inference
            action_probs, _ = self.model(state)  # Obtain action probabilities from the model

        distribution = Categorical(action_probs)  # Create a categorical distribution based on action probabilities
        
        if self.mode == 'train':
            action = distribution.sample()  # Sample an action during training
        elif self.mode == 'eval':
            action = action_probs.argmax(dim=-1)  # Select the action with the highest probability during evaluation
        
        log_prob = distribution.log_prob(action)  # Calculate the log probability of the selected action
        confidence = action_probs[0, action.item()].item()  # Extract the selected action's probability (confidence)

        return action.item(), log_prob.item(), confidence

    # Method to save the current state of the agent to a file
    def save_state(self, filename):
        # Create a dictionary to hold relevant state information and parameters
        agent_state = {
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon,
            'gae_lambda': self.gae_lambda,
            'model_state_dict': self.model.state_dict(),  # Save the model's state dictionary
            'optimizer_state_dict': self.optimizer.state_dict(),  # Save the optimizer's state dictionary
        }
        # Open a file in write-binary mode and dump the agent_state dictionary using pickle
        with open(filename, 'wb') as f:
            pickle.dump(agent_state, f)

    # Method to load the agent state from a file
    def load_state(self, filename):
        # Open the file in read-binary mode and load the agent state dictionary using pickle
        with open(filename, 'rb') as f:
            agent_state = pickle.load(f)
            # Update the agent's attributes with the loaded state
            self.input_dim = agent_state['input_dim']
            self.action_dim = agent_state['action_dim']
            self.lr_actor = agent_state['lr_actor']
            self.lr_critic = agent_state['lr_critic']
            self.gamma = agent_state['gamma']
            self.clip_epsilon = agent_state['clip_epsilon']
            self.gae_lambda = agent_state['gae_lambda']
            self.model.load_state_dict(agent_state['model_state_dict'])  # Load model parameters
            self.optimizer.load_state_dict(agent_state['optimizer_state_dict'])  # Load optimizer state
 
    # Define a method for computing Generalized Advantage Estimation (GAE)
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        # Flatten the 'values' tensor to ensure it's a 1D tensor
        values = values.squeeze()
        
        # Ensure 'next_value' is a 1D tensor with a single value
        next_value = next_value.squeeze()
        
        # Concatenate 'values' with 'next_value' for GAE calculation
        # 'next_value' is unsqueezed if it's a scalar, making it a 1D tensor with a single element
        values = torch.cat([values, next_value.unsqueeze(0) if next_value.dim()==0 else next_value], dim=0)
        
        # Initialize the variable for accumulated GAE
        gae = 0
        # Initialize a list to store computed returns
        returns = []
        # Loop through rewards in reverse order to calculate GAE
        for step in reversed(range(len(rewards))):
            # Calculate the delta for GAE computation
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            # Update GAE using delta, gamma, tau, and masks
            gae = delta + gamma * tau * masks[step] * gae
            # Insert the computed return (GAE + value) at the beginning of the returns list
            returns.insert(0, gae + values[step])
        # Return the list of computed returns
        return returns

    # Define a method for updating the policy and value networks using collected experience and PPO loss
    def update(self, states, actions, rewards, next_states, dones, log_probs_old, advantage_estimates, returns):
        # Convert inputs to PyTorch tensors if they are not already tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        advantage_estimates = torch.tensor(advantage_estimates, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize the advantage estimates
        advantages = (advantage_estimates - advantage_estimates.mean()) / (advantage_estimates.std() + 1e-8)

        # Obtain action probabilities, log probabilities of the taken actions, and state values from the current policy
        action_probs, state_values = self.model(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        # Calculate entropy of the action distribution for entropy regularization
        entropy_loss = dist.entropy().mean()

        # Compute the ratio of the new policy probability to the old policy probability for the taken actions
        ratios = torch.exp(log_probs - log_probs_old.detach())

        # Compute the surrogate loss components for the clipped PPO objective
        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
        # Calculate the policy loss including the entropy bonus
        policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_loss

        # Calculate the value loss using mean squared error
        value_loss = F.mse_loss(state_values.squeeze(), returns)

        # Calculate the total loss as the sum of policy loss and scaled value loss
        total_loss = policy_loss + 0.5 * value_loss

        # Zero gradients, perform backpropagation on total loss, clip gradients, then perform an optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

    # Define a method to process a batch of experiences
    def process_batch(self, batch):
        # Unpack the batch into separate lists for each component of an experience
        states, actions, rewards, next_states, dones, log_probs = zip(*batch)
        
        # Convert each list of numpy arrays for more efficient processing
        states_np = np.array(states)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards)
        next_states_np = np.array(next_states)
        dones_np = np.array(dones)
        log_probs_np = np.array(log_probs)

        # Convert numpy arrays to PyTorch tensors for neural network processing
        states = torch.FloatTensor(states_np)
        actions = torch.LongTensor(actions_np)
        rewards = torch.FloatTensor(rewards_np)
        next_states = torch.FloatTensor(next_states_np)
        dones = torch.FloatTensor(dones_np)
        log_probs = torch.FloatTensor(log_probs_np)

        # Compute the value of the next state using the model, without gradient computation
        with torch.no_grad():
            next_state_values = self.model(next_states)[-1].squeeze(-1)
        
        # Compute returns using Generalized Advantage Estimation (GAE)
        returns = self.compute_gae(next_state_values, rewards, 1 - dones, self.model(states)[-1].squeeze(-1))

        # Convert returns to a tensor, if not already
        returns = torch.FloatTensor(returns)

        # Detach returns tensor from the computation graph if it requires gradients
        if isinstance(returns, torch.Tensor) and returns.requires_grad:
            returns = returns.detach()

        # Compute advantages by subtracting the value of the current states from returns
        advantages = returns - self.model(states)[-1].squeeze(-1)

        # Return processed tensors to be used for learning
        return states, actions, returns, advantages, log_probs, rewards, next_states, dones

    # Define a method for learning from a batch of experiences
    def learn(self, batch):
        # Process the batch to extract tensors for states, actions, rewards, etc.
        states, actions, returns, advantages, log_probs, rewards, next_states, dones = self.process_batch(batch)
        
        # Compute the value of the next states, ensuring no gradients are computed
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
            next_state_values = next_state_values.squeeze(-1)

        # Recompute returns using GAE for consistency and to ensure tensors are updated
        returns = self.compute_gae(next_state_values, rewards, 1 - dones, self.model(states)[-1].squeeze(-1))
        
        # Update the model using the processed batch and computed returns and advantages
        self.update(states, actions, rewards, next_states, dones, log_probs, advantages, returns)

# Define a function to evaluate the agent on out-of-sample data
def validate_agent(agent_save_filename, testing_set):
        # Initialize the trading environment with out-of-sample data
        env_out_of_sample = ForexTradingEnv(testing_set)

        observation_space_shape = env_out_of_sample.observation_space.shape[0]  # Or any other method to find the total features per timestep.

        input_dim = observation_space_shape

        # Determine the action space size from the environment
        action_size = env_out_of_sample.action_space.n

        # Create a new PPOAgent instance for evaluation purposes
        agent_evaluation = PPOAgent(input_dim, action_size)

        # Load the saved agent state for evaluation
        agent_evaluation.load_state(agent_save_filename)

        # Switch the agent's model to evaluation mode
        agent_evaluation.set_eval_mode()

        # Print a message indicating the start of out-of-sample testing
        print("Transitioning to validate testing...")

        # Initialize the done flag to False
        done = False

        # Initialize the total reward for out-of-sample testing
        total_test_reward = 0

        # Initialize a list to store all observations during out-of-sample testing
        all_observations = []

        # Reset the environment to get the initial state for out-of-sample testing
        state = env_out_of_sample.reset()

        # Continue the loop until the episode ends (the 'done' flag is True)
        while not done:
            # Select an action based on the current state using the agent's policy
            action, _, confidence = agent_evaluation.select_action(state)
            # Temporary storage for out-of-sample trade history
            out_of_sample_trade_history = env_out_of_sample.trade_history
            # Execute the selected action in the environment and observe the next state and reward
            next_state, reward, done, _ = env_out_of_sample.step(action, out_of_sample_trade_history, confidence)
            # Accumulate the total reward for this episode
            total_test_reward += reward
            # Record the next state for later analysis
            all_observations.append(next_state)

            # Update the current state to the next state to continue the loop
            state = next_state

        # Evaluate performance
        performance_metrics_out_of_sample = env_out_of_sample.evaluate_performance()

        final_profit_evaluation = {performance_metrics_out_of_sample['Final Profit']}

        # Return the final profit from the out-of-sample evaluation as a float, the environment instance, and the total test reward for further analysis or use
        return total_test_reward, final_profit_evaluation

def train_agent_in_sample(episodes, training_set, testing_set):
        # Instantiate the trading environment with in-sample data
        env_in_sample = ForexTradingEnv(training_set)
        # Initialize the best validation score to a high number (effectively infinity)
        best_validation_score = float('inf')
        episodes_without_improvement = 0
        patience = 10  # Number of episodes to tolerate no improvement in validation score

        # Initialize a list to store observations collected during training
        all_observations = []

        observation_space_shape = env_in_sample.observation_space.shape[0]  # Or any other method to find the total features per timestep.

        input_dim = observation_space_shape

        # Initialize the PPO agent with the specified input dimension and action space size
        agent = PPOAgent(input_dim, 4)

        # Initialize a variable to keep track of the total number of steps taken during training
        total_steps = 0

        # Loop through each episode for training the agent
        for e in range(episodes):
            print(f'On episode {e}')
            batch = []  # Start with an empty batch for aggregating experiences

            # Reset the environment to get the initial state and store it
            state = env_in_sample.reset()
            all_observations.append(state)

            # Ensure the agent's model is set to training mode
            agent.set_train_mode()
            # Reset the environment again for consistency
            state_in_sample = env_in_sample.reset()
            # Initialize total reward for the episode
            total_reward = 0
            # Flag to indicate if the episode is done
            done = False

            # Skip episodes with NaN values in the in-sample data
            if training_set.isnull().values.any():
                print(f"Skipping episode {e+1} due to NaN values in in-sample data.")
                continue

            # Main loop to interact with the environment until the episode ends
            while not done:
                # Agent selects an action based on the current state
                action, log_prob, confidence = agent.select_action(state)

                # Fetch the current trade history from the environment
                in_sample_trade_history = env_in_sample.trade_history

                # Apply the selected action to the environment and receive feedback
                next_state, reward, done, _ = env_in_sample.step(action, in_sample_trade_history, confidence)
                # Add experience to batch
                batch.append((state, action, reward, next_state, done, log_prob))
                # Store the next observation
                all_observations.append(next_state)
                # Accumulate the total reward
                total_reward += reward

                # Increment the total steps counter
                total_steps += 1

                # Update the state for the next iteration
                state = next_state
                # Desired batch size for training the agent
                desired_batch_size = 51
                # If the batch size exceeds the desired size, learn from the batch and reset it
                if len(batch) > desired_batch_size:
                    # Train the agent using the collected batch and reset the batch
                    agent.learn(batch)
                    batch = []

            agent_save_filename = os.path.join(f"agent_state.pkl")
                
            agent.save_state(agent_save_filename)  # Save the current best model state to the specified path.

            # Evaluate the agent's performance based on in-sample data
            performance_metrics_in_sample = env_in_sample.evaluate_performance()

            # In sample profit
            total_in_sample_profit = performance_metrics_in_sample['Final Profit']

            total_test_reward, total_out_of_sample_profit = validate_agent(agent_save_filename, testing_set)

            print(f'epoch {e} total training reward is {total_reward} and toal profit is {total_in_sample_profit} and total validation reward is {total_test_reward} and total profit is {total_out_of_sample_profit}')

            if total_test_reward < best_validation_score:  # Check if the current validation score is better (lower) than the best recorded score.
                best_validation_score = total_test_reward  # Update the best validation score to the current score.
                episodes_without_improvement = 0  # Reset the counter for episodes without improvement.

                # Define the path and filename where the current best model state will be saved.
                agent_save_filename_best = os.path.join(f"agent_state_best.pkl")
                
                agent.save_state(agent_save_filename_best)  # Save the current best model state to the specified path.
                
                # This line indicates where you could add additional code to save the model state, if needed.
            else:  # If the current validation score is not better than the best recorded score,
                episodes_without_improvement += 1  # Increment the counter for episodes without improvement.
                
            if episodes_without_improvement >= patience:  # Check if the episodes without improvement exceed the patience threshold.
                print("Early stopping triggered")  # Print a message indicating that early stopping is triggered.
                break  # Exit the training loop.

def evaluate_model(agent_save_filename, testing_set):
        # Initialize the trading environment with out-of-sample data
        env_out_of_sample = ForexTradingEnv(testing_set)

        observation_space_shape = env_out_of_sample.observation_space.shape[0]  # Or any other method to find the total features per timestep.

        input_dim = observation_space_shape

        # Determine the action space size from the environment
        action_size = env_out_of_sample.action_space.n

        # Create a new PPOAgent instance for evaluation purposes
        agent_evaluation = PPOAgent(input_dim, action_size)

        # Load the saved agent state for evaluation
        agent_evaluation.load_state(agent_save_filename)

        # Switch the agent's model to evaluation mode
        agent_evaluation.set_eval_mode()

        # Initialize the total reward for out-of-sample testing
        total_test_reward = 0

        # Initialize a list to store all observations during out-of-sample testing
        all_observations = []

        # Reset the environment to get the initial state for out-of-sample testing
        state = env_out_of_sample.reset()
        # Append the initial state to the observations list
        all_observations.append(state)

        done = False

        # Continue the loop until the episode ends (the 'done' flag is True)
        while not done:
            # Select an action based on the current state using the agent's policy
            action, _, confidence = agent_evaluation.select_action(state)
            # Temporary storage for out-of-sample trade history
            out_of_sample_trade_history = env_out_of_sample.trade_history

            # Execute the selected action in the environment and observe the next state and reward
            next_state, reward, done, _ = env_out_of_sample.step(action, out_of_sample_trade_history, confidence)
            # Accumulate the total reward for this episode
            total_test_reward += reward
            # Record the next state for later analysis
            all_observations.append(next_state)

            # Update the current state to the next state to continue the loop
            state = next_state

        # Evaluate performance
        performance_metrics_out_of_sample = env_out_of_sample.evaluate_performance()
                
        # Assuming equity curve plotting is desired at the end of each episode
        plt.figure(figsize=(15, 10))  # Create a figure with specified size
        ax = plt.gca()  # Get the current axes for plotting

        # Get the current date and time as a string for potential use in titles or filenames
        current_datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Extract and plot the equity curve based on closed balances and corresponding dates
        balances = []
        dates = []
        for trade in env_out_of_sample.trade_history:
            if 'Closed Balance' in trade:
                balances.append(trade['Closed Balance'])
                dates.append(datetime.strptime(trade['Date'], '%Y-%m-%d'))

        ax.plot(dates, balances, '-o', label='Equity Curve')  # Plot the equity curve with markers

        # Customize the plot with titles, labels, and formatting
        ax.set_title(f'Equity Curve for {total_test_reward}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Balance')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the x-axis dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Auto-format date labels for readability

        # Prepare data for a table summarizing key performance metrics
        metrics_data = [
            ['Final Balance', f"{performance_metrics_out_of_sample['Final Balance']:.2f}"],
            ['Final Profit', f"{performance_metrics_out_of_sample['Final Profit']:.2f}"],
            ['Maximum Drawdown', performance_metrics_out_of_sample['Maximum Drawdown']],
            ['Highest Daily Loss', performance_metrics_out_of_sample['Highest Daily Loss']],
            ['Profit Factor', performance_metrics_out_of_sample['Profit Factor']],
            ['Number of Closed Trades', str(performance_metrics_out_of_sample['Number of Closed Trades'])],
            ['Percent Profitable', performance_metrics_out_of_sample['Percent Profitable']],
            ['Total Reward', total_test_reward]
        ]

        # Add the table to the plot, specifying its location and appearance
        table = plt.table(cellText=metrics_data, colLabels=['Metric', 'Value'], loc='bottom', cellLoc='center', bbox=[0.0, -0.5, 1.0, 0.3])
        table.auto_set_font_size(False)  # Disable automatic font size setting
        table.set_fontsize(9)  # Set font size for the table
        table.scale(1, 1.5)  # Scale the table to fit better in the plot

        # Adjust the layout to make room for the table
        plt.subplots_adjust(left=0.2, bottom=0.2, top=0.583, right=0.567)

        # Check if the observations directory exists, create it if not
        specific_folder_name_validation = "Evaluation"
        if not os.path.exists(specific_folder_name_validation):
            os.makedirs(specific_folder_name_validation)

        # Construct the filename using the total reward and the unique identifier
        plot_filename_out_of_sample = os.path.join(specific_folder_name_validation, f"Evaluation_EquityCurve.png")

        # Save the plot to the specified file
        plt.savefig(plot_filename_out_of_sample)
        plt.close()  # Close the plot to free up system resources

        # Clear the current figure to avoid any overlap with future plots
        plt.clf()

        # Construct the filename for saving performance metrics, including the unique identifier and the current datetime
        metrics_filename = os.path.join(specific_folder_name_validation, f"Evaluation_PerformanceMetrics.txt")

        # Open the file for writing and record each performance metric
        with open(metrics_filename, 'w') as file:
            for metric in metrics_data:
                key, value = metric
                file.write(f"{key}: {value}\n")  

        # Construct the filename for saving the out-of-sample trade history, including the currency pair, timeframe, evaluation dates, and episode number
        out_of_sample_filename = os.path.join(specific_folder_name_validation, f"Evaluation_trade_history_episode.csv")

        # Convert the list of dictionaries (trade history) into a pandas DataFrame
        df_out_of_sample = pd.DataFrame(env_out_of_sample.trade_history)

        # Save the DataFrame to a CSV file at the specified path
        df_out_of_sample.to_csv(out_of_sample_filename, index=False)

# Define a function to fetch and prepare FX data from MT5 for a given symbol and timeframe
def fetch_fx_data_mt5(symbol, timeframe_str, start_date, end_date):

    # Define your MetaTrader 5 account number
    account_number = 1520146991
    # Define your MetaTrader 5 password
    password = 'HWrpmybVOs9*'
    # Define the server name associated with your MT5 account
    server_name ='FTMO-Demo2'

    # Initialize MT5 connection; if it fails, print error message and exit
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # Attempt to log in with the given account number, password, and server
    authorized = mt5.login(account_number, password=password, server=server_name)
    # If login fails, print error message, shut down MT5 connection, and exit
    if not authorized:
        print("login failed, error code =", mt5.last_error())
        mt5.shutdown()
        quit()
    # On successful login, print a confirmation message
    else:
        print("Connected to MetaTrader 5")

    # Set the timezone to Berlin, as MT5 times are in UTC
    timezone = pytz.timezone("Europe/Berlin")

    # Convert start and end dates to datetime objects, considering the timezone
    start_date = start_date.replace(tzinfo=timezone)
    end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=timezone)

    # Define a mapping from string representations of timeframes to MT5's timeframe constants
    timeframes = {
        '1H': mt5.TIMEFRAME_H1,
        'DAILY': mt5.TIMEFRAME_D1,
        '12H': mt5.TIMEFRAME_H12,
        '2H': mt5.TIMEFRAME_H2,
        '3H': mt5.TIMEFRAME_H3,
        '4H': mt5.TIMEFRAME_H4,
        '6H': mt5.TIMEFRAME_H6,
        '8H': mt5.TIMEFRAME_H8,
        '1M': mt5.TIMEFRAME_M1,
        '10M': mt5.TIMEFRAME_M10,
        '12M': mt5.TIMEFRAME_M12,
        '15M': mt5.TIMEFRAME_M15,
        '2M': mt5.TIMEFRAME_M2,
        '20M': mt5.TIMEFRAME_M20,
        '3M': mt5.TIMEFRAME_M3,
        '30M': mt5.TIMEFRAME_M30,
        '4M': mt5.TIMEFRAME_M4,
        '5M': mt5.TIMEFRAME_M5,
        '6M': mt5.TIMEFRAME_M6,
        '1MN': mt5.TIMEFRAME_MN1,
        '1W': mt5.TIMEFRAME_W1
    }

    # Retrieve the MT5 constant for the requested timeframe
    timeframe = timeframes.get(timeframe_str)
    # If the requested timeframe is invalid, print error message, shut down MT5, and return None
    if timeframe is None:
        print(f"Invalid timeframe: {timeframe_str}")
        mt5.shutdown()
        return None

    # Fetch the rates for the given symbol and timeframe within the start and end dates
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    # If no rates were fetched, print error message, shut down MT5, and return None
    if rates is None:
        print("No rates retrieved, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    
    # Convert the fetched rates into a Pandas DataFrame
    rates_frame = pd.DataFrame(rates)
    # Convert the 'time' column from UNIX timestamps to human-readable dates
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # Set the 'time' column as the DataFrame index and ensure its format is proper for datetime
    rates_frame.set_index('time', inplace=True)
    rates_frame.index = pd.to_datetime(rates_frame.index, format="%Y-%m-%d %H:%M:%S")

    # Check if 'tick_volume' column is present in the fetched data
    if 'tick_volume' not in rates_frame.columns:
        print("tick_volume is not in the fetched data. Ensure it's included in the API call.")
    else:
        print("tick_volume is included in the data.")
    
    # Shut down the MT5 connection before returning the data
    mt5.shutdown()
    
    # Return the prepared DataFrame containing the rates
    return rates_frame

# Defines a function to prompt the user for a date input and ensure it's in the correct format
def get_user_date_input(prompt):
    # Specify the expected date format
    date_format = '%Y-%m-%d'
    # Prompt the user for a date
    date_str = input(prompt)
    # Loop until a valid date format is entered
    while True:
        try:
            # Attempt to parse the date; if successful, it's valid, so return it
            pd.to_datetime(date_str, format=date_format)
            return date_str
        except ValueError:
            # If parsing fails, notify the user and prompt again
            print("The date format is incorrect. Please enter the date in 'YYYY-MM-DD' format.")
            date_str = input(prompt)

def calculate_indicators(data, bollinger_length=12, bollinger_std_dev=1.5, sma_trend_length=50, window=9):
    print("Doing calculate_indicators function")
    # Calculate the 50-period simple moving average of the 'close' price
    data['SMA_50'] = ta.sma(data['close'], length=50)
    # Calculate the 200-period simple moving average of the 'close' price
    data['SMA_200'] = ta.sma(data['close'], length=200)
    
    # Calculate the 50-period exponential moving average of the 'close' price
    data['EMA_50'] = ta.ema(data['close'], length=50)
    # Calculate the 200-period exponential moving average of the 'close' price
    data['EMA_200'] = ta.ema(data['close'], length=200)

    # Calculate the 9-period exponential moving average for scalping strategies
    data['EMA_9'] = ta.ema(data['close'], length=9)
    # Calculate the 21-period exponential moving average for scalping strategies
    data['EMA_21'] = ta.ema(data['close'], length=21)
    
    # Generate original Bollinger Bands with a 20-period SMA and 2 standard deviations
    original_bollinger = ta.bbands(data['close'], length=20, std=2)
    # The 20-period simple moving average for the middle band
    data['SMA_20'] = ta.sma(data['close'], length=20)
    # Upper and lower bands from the original Bollinger Bands calculation
    data['Upper Band'] = original_bollinger['BBU_20_2.0']
    data['Lower Band'] = original_bollinger['BBL_20_2.0']

    # Generate updated Bollinger Bands for scalping with custom length and standard deviation
    updated_bollinger = ta.bbands(data['close'], length=bollinger_length, std=bollinger_std_dev)
    # Assign lower, middle, and upper bands for scalping
    data['Lower Band Scalping'], data['Middle Band Scalping'], data['Upper Band Scalping'] = updated_bollinger['BBL_'+str(bollinger_length)+'_'+str(bollinger_std_dev)], ta.sma(data['close'], length=bollinger_length), updated_bollinger['BBU_'+str(bollinger_length)+'_'+str(bollinger_std_dev)]
    
    # Calculate the MACD indicator and its signal line
    macd = ta.macd(data['close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal_Line'] = macd['MACDs_12_26_9']
    
    # Calculate the Relative Strength Index (RSI) with the specified window length
    data[f'RSI_{window}'] = ta.rsi(data['close'], length=window).round(2)

    # Calculate a 5-period RSI for scalping strategies
    data[f'RSI_5 Scalping'] = ta.rsi(data['close'], length=5).round(2)

    # Calculate a simple moving average for trend analysis in scalping strategies
    data[f'SMA_{sma_trend_length}'] = ta.sma(data['close'], length=sma_trend_length)

    # Calculate the Stochastic Oscillator
    stoch = ta.stoch(data['high'], data['low'], data['close'])
    data['Stoch_%K'] = stoch['STOCHk_14_3_3']
    data['Stoch_%D'] = stoch['STOCHd_14_3_3']

    # Return the data with added indicators
    return data

def split_and_save_dataset(dataset, timeframe, pair):
    # Calculate the split index for training and testing
    split_index_train_test = int(len(dataset) * 0.8)
    
    # Split the dataset into training and testing
    training_set = dataset.iloc[:split_index_train_test]
    testing_set = dataset.iloc[split_index_train_test:]
    
    # Further split the training set to create a validation set (80% training, 20% validation of the training set)
    split_index_train_val = int(len(training_set) * 0.8)
    final_training_set = training_set.iloc[:split_index_train_val]
    validation_set = training_set.iloc[split_index_train_val:]
    
    # Save the sets into CSV files
    final_training_set.to_csv(f'training_{pair}_{timeframe}_data.csv', index=True)
    validation_set.to_csv(f'validation_{pair}_{timeframe}_data.csv', index=True)
    testing_set.to_csv(f'testing_{pair}_{timeframe}_data.csv', index=True)
    
    # Return the split datasets
    return final_training_set, validation_set

def read_csv_to_dataframe(file_path):
    # Read the CSV file into a DataFrame and set the first column as the index
    df = pd.read_csv(file_path)
    # Set the 'time' column as the DataFrame index and ensure its format is proper for datetime
    df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    return df

def training_ppo_model(choice):
    if choice == '1':
        # Retrieve and store the current date
        current_date = str(datetime.now().date())
        current_date_datetime = datetime.now().date()

        # Calculate the date for one month before the current date
        one_month_before = current_date_datetime - relativedelta(months=1)
        # Convert to string if needed
        one_month_before_str = str(one_month_before)

        # Hardcoded start date for strategy evaluation
        strategy_start_date_all = "1971-01-04"
        # Use the current date as the end date for strategy evaluation
        strategy_end_date_all = current_date

        # Convert string representation of dates to datetime objects for further processing
        start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
        end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

        # Prompt the user to decide if they want real-time data updates and store the boolean result
        #enable_real_time = input("Do you want to enable real-time data updates? (yes/no): ").lower().strip() == 'yes'

        # Prompt the user for the desired timeframe for analysis and standardize the input
        timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()
        # Prompt the user for the currency pair they're interested in and standardize the input
        Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

        # Ask the user for start and end date for training
        training_start_date = one_month_before_str
        training_end_date = current_date

        # Fetch and prepare the FX data for the specified currency pair and timeframe
        eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

        # Apply technical indicators to the data using the 'calculate_indicators' function
        eur_usd_data = calculate_indicators(eur_usd_data) 

        # Filter the EUR/USD data for the in-sample training period
        dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

        dataset = dataset.fillna(0)

        training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)
    elif choice == '3':
        testing_set = read_csv_to_dataframe('testing_EURUSD_1H_data.csv')
        training_set = read_csv_to_dataframe('training_EURUSD_1H_data.csv')

    episodes = 10

    train_agent_in_sample(episodes, training_set, testing_set)

def evaluate_ppo_model():
    evaluation_dataset = read_csv_to_dataframe('testing_EURUSD_1H_data.csv')

    evaluate_model("agent_state_best.pkl", evaluation_dataset)

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Train a PPO model with latest data")
        print("2 - Evaluate a PPO model")
        print("3 - Train a PPO model with csv file data")
        print("4 - Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            training_ppo_model(choice)
            break
        elif choice == '2':
            evaluate_ppo_model()
            break
        elif choice == '3':
            training_ppo_model(choice)
            break
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()