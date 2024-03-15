from __future__ import print_function
import os
import copy
import random
from collections import deque
current_path = os.getcwd()
from timeit import default_timer as t
from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.vector import Vector

TRAIN = {'y':True,'n':False}[input("Train RL agent? y/n ")]
GUI = False
if TRAIN == False:
    GUI = {'y':True,'n':False}[input('Would you like a GUI? y/n ')]
    HUMAN_PLAYER = {'y':True,'n':False}[input("Human player? y/n ")]
    RL_AGENT_AS_MRS_PEACOCK = {'y':True,'n':False}[input("Should Mrs. Peacock use reinforcement learning? *Warning: this feature is highly experimental* --->y/n ")]  

# Kvlang string to create widgets and binding for GUI
kv = """

<GamePiece1>:
    size_hint: .025, .03
    canvas:
        Color:
            rgb: 1, 0, 0
        Ellipse:
            pos: self.pos
            size: self.size
<GamePiece2>:
    size_hint: .025, .03
    canvas:
        Color:
            rgb: 1, 1, 0
        Ellipse:
            pos: self.pos
            size: self.size
<GamePiece3>:
    size_hint: .025, .03
    canvas:
        Color:
            rgb: 1, 1, 1 
        Ellipse:
            pos: self.pos
            size: self.size
<GamePiece4>:
    size_hint: .025, .03
    canvas:
        Color:
            rgb: 0, .6, 0 
        Ellipse:
            pos: self.pos
            size: self.size
<GamePiece5>:
    size_hint: .025, .03
    canvas:
        Color:
            rgb: 0, .4, .8
        Ellipse:
            pos: self.pos
            size: self.size
<GamePiece6>:
    size_hint: .025, .03
    canvas:
        Color:
            rgb: .4, 0, .8
        Ellipse:
            pos: self.pos
            size: self.size
            
<MainWidget>:
    piece_red: scarlet
    piece_yellow: mustard
    piece_white: white
    piece_green: green
    piece_blue: blue
    piece_purple: purple
    
    FloatLayout:
        id: board
        canvas:
            Rectangle:
                size: root.size
                source: "clue_board2.png"
    
    GamePiece1:
        id: scarlet
        pos: self.new_pos
            
    GamePiece2:
        id: mustard
        pos: self.new_pos
        
    GamePiece3:
        id: white
        pos: self.new_pos
            
    GamePiece4:
        id: green
        pos: self.new_pos
        
    GamePiece5:
        id: blue
        pos: self.new_pos
            
    GamePiece6:
        id: purple
        pos: self.new_pos

            
        
"""

# load Kvlang string to build layout
Builder.load_string(kv)

#Ask user questions to determine what to do
ROUNDS = 1
SETS_OF_ROUNDS = 1
OLD_LOGIC = {'y':True,'n':False}[input("Use broken logic for movement? y/n ")]
 
if TRAIN: 
    HUMAN_PLAYER,DISPLAY_STATS = False,False
    SETS_OF_ROUNDS = int(input('How many sets of rounds?')) + 1
    RL_AGENT_AS_MRS_PEACOCK = {'y':True,'n':False}[input("Train rl with its own data? (Mrs peacock as RL agent) ---> y/n ")]
if HUMAN_PLAYER == False:
    if TRAIN == False:
        DISPLAY_STATS = {'y':True,'n':False}[input("Print out moves? y/n ")]
    if GUI == False:
        ROUNDS = int(input("How Many Rounds?"))
else: DISPLAY_STATS = True
if TRAIN or RL_AGENT_AS_MRS_PEACOCK:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import warnings
    warnings.filterwarnings('ignore')
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import Model, Input
    dtype = 'float16'
    for GPU in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(GPU, True)
    tf.keras.mixed_precision.set_global_policy(f'mixed_{dtype}')#'mixed_float16''float32' mixed_
if RL_AGENT_AS_MRS_PEACOCK:
    HYBRID_EUCLIDEAN_PEACOCK = input("Should Mrs Peacock consider distance to doors in addition to percieved value?")
    if RL_AGENT_AS_MRS_PEACOCK:
            try:
                model = tf.keras.models.load_model(f'{current_path}/clue1.h5')
                #model.compile(loss='mse',optimizer='adam',metrics=['mse'])
            except Exception as e:
                print(f"Error! No model has been trained. Please either run this script again with Train RL agent as true or make sure clue1.h5 is in {current_path}/clue1.h5)")
                print(e)
                exit()
    
class Player:
    def __init__(self, name, current_pos=[], hand=[], who=[], where=[], weapon=[],accuse_threshold=.75,num_players=6,all_cards=[],player_type='logic'):
        self.name = name
        self.current_pos = current_pos
        self.numplayers = num_players
        self.hand = hand
        self.history = current_pos
        self.loser = False
        self.moves_left = 0
        self.earned_rewards = 0
        self.player_type = player_type
        self.conf = {}
        self.case_file_who_prob = [i for i in who]
        self.case_file_where_prob = [i for i in where]
        self.case_file_weapon_prob = [i for i in weapon]
        self.room = ''
        self.certainty = 0
        self.accuse_threshold = accuse_threshold
        self.ready_accuse = False
        self.all_players_possible_cards = {}
        self.stored_player_guesses = {}#dict with nested lists of guesses
        for name in who:
            self.stored_player_guesses[name] = []
            self.all_players_possible_cards[name] = [i for i in all_cards]
    def update_casefile_probs(self,guess_maker,cards,guess_answers,passed_interogations,answeree):
        
        if len(self.all_players_possible_cards[self.name]) > 4:#account for cards in your own hand
            for other_player_names in self.all_players_possible_cards:
                if other_player_names != self.name:
                    for your_cards_in_hand in self.hand:
                        if your_cards_in_hand in self.all_players_possible_cards[other_player_names]:
                            self.all_players_possible_cards[other_player_names].remove(your_cards_in_hand)
                            self.earned_rewards += 1
            for your_cards_in_hand in self.hand:
                if your_cards_in_hand in self.case_file_weapon_prob:
                    self.case_file_weapon_prob.remove(your_cards_in_hand)
                    self.earned_rewards += 1
                if your_cards_in_hand in self.case_file_where_prob:
                    self.case_file_where_prob.remove(your_cards_in_hand)
                    self.earned_rewards += 25
                if your_cards_in_hand in self.case_file_who_prob:
                    self.case_file_who_prob.remove(your_cards_in_hand)
                    self.earned_rewards += 1
            self.all_players_possible_cards[self.name] = self.hand#deal with these for guesses
            
        #begin of section coded by Talon, edited by John to track guesses/rewards.
        if len(passed_interogations) > 0: #remove all guess cards from decks which say they have none of the three
            for player_names in passed_interogations:
                for card in cards:
                    if card in self.all_players_possible_cards[player_names]:
                        self.all_players_possible_cards[player_names].remove(card)
                        self.earned_rewards += 1
                    for stored_guesses in self.stored_player_guesses[player_names]: #also account for these wash guesses and eliminate fro mstored guesses for the wash player
                        if card in stored_guesses:
                            stored_guesses.remove(card)
                            self.earned_rewards += 1

        if guess_maker == self.name:
            if len(guess_answers) > 0:
                for player_name in self.all_players_possible_cards:
                    if player_name != answeree:
                        if guess_answers[0] in self.all_players_possible_cards[player_name]:
                            self.all_players_possible_cards[player_name].remove(guess_answers[0])
                            self.earned_rewards += 1
                        if guess_answers[0] in self.case_file_who_prob:
                            self.case_file_who_prob.remove(guess_answers[0])
                            self.earned_rewards += 1
                        if guess_answers[0] in self.case_file_weapon_prob:
                            self.case_file_weapon_prob.remove(guess_answers[0])
                            self.earned_rewards += 1
                        if guess_answers[0] in self.case_file_where_prob:
                            self.case_file_where_prob.remove(guess_answers[0])
                            self.earned_rewards += 1
        #End of section written by tallon:
        
        if len(guess_answers) != 0:  
            if cards not in self.stored_player_guesses[answeree]:
                self.stored_player_guesses[answeree].append(cards) 
                self.earned_rewards += -1
        
        for name in self.stored_player_guesses:
            for appended_guess in self.stored_player_guesses[name]:
                if len(appended_guess)==1:
                    for other_player_names in self.all_players_possible_cards:
                        if other_player_names != name:
                            if appended_guess[0] in self.all_players_possible_cards[other_player_names]:
                                self.all_players_possible_cards[other_player_names].remove(appended_guess[0])
                                self.earned_rewards += 1
                    if appended_guess[0] in self.case_file_who_prob:
                        self.case_file_who_prob.remove(appended_guess[0])
                        self.earned_rewards += 1
                    if appended_guess[0] in self.case_file_weapon_prob:
                        self.case_file_weapon_prob.remove(appended_guess[0])
                        self.earned_rewards += 1
                    if appended_guess[0] in self.case_file_where_prob:
                        self.case_file_where_prob.remove(appended_guess[0])
                        self.earned_rewards += 1
                         
        big_list = []
        for players in self.all_players_possible_cards:
            for cards in self.all_players_possible_cards[players]:
                big_list.append(cards)
        for card in self.case_file_who_prob:
            if card not in big_list:
                self.case_file_who_prob = [card]
        for card in self.case_file_where_prob:
            if card not in big_list:
                self.case_file_where_prob = [card]
        for card in self.case_file_weapon_prob:
            if card not in big_list:
                self.case_file_weapon_prob = [card]        
    def dice_roll(self):
        
        """Roles dice and resets history"""
        self.moves_left = random.randint(1, 6) + random.randint(1, 6)
        
        return self.moves_left
    def guessing(self, guess):

        who_card, where_card, weapon_card = guess
        one_guess = []
        if who_card in self.hand:
            one_guess.append(who_card)
        if where_card in self.hand:
            one_guess.append(where_card)
        if weapon_card in self.hand:
            one_guess.append(weapon_card)

        if len(one_guess)>1:
            one_guess = [random.choice(one_guess)]
            
        else:
            pass

        return one_guess
    def check_ready_to_accuse(self):
        """Get max probability value from each probability dictionary for player calling this function, compare to threshold"""
        #self.update_casefile_probs(guess_maker=[],cards=[],guess_answers=[],passed_interogations=[],answeree=[])
        if len(self.case_file_who_prob)<1 or len(self.case_file_weapon_prob)<1 or len(self.case_file_where_prob)<1:
            print(len(self.case_file_who_prob))
            print(f"Freezing program. There are not enough choices to make a guess: guess options are {self.case_file_who_prob,self.case_file_where_prob,self.case_file_weapon_prob}")
            quit()
        
        #If this crashes, then some logic has and error which deleted the confidential card from the possible
        self.certainty = (1/len(self.case_file_who_prob))*(1/len(self.case_file_where_prob))*(1/len(self.case_file_weapon_prob))
        
        if self.certainty >= self.accuse_threshold:
            self.ready_accuse = True          
    def accuse(self, guess = [],confidential=[],p=False):
        """Input guess [who_card, where_card, weapon_card] and confidential [who_card, where_card, weapon_card]

        Updates current_player.ready_accuse
        
        Returns: True or False and sets char to winner or loser if ready_accuse = True"""
        if guess == []:
                guess = [random.choice(self.case_file_who_prob), random.choice(self.case_file_where_prob), random.choice(self.case_file_weapon_prob)]
        self.check_ready_to_accuse()

        if self.ready_accuse: 
            if guess == confidential:
                if p:
                    print("Accusation:", guess, ". Case File: ", f"{clue_game.confidential}.",current_player.name, "is the winner!")
                    self.earned_rewards += 100
                return True
            else:
                self.loser = True
                if p:
                    print(current_player.name, "has lost!")
                    self.earned_rewards -= 100
            return False       
class setup_board:
    def __init__(self,current_pos = {'Mr. Green' : [25,10],'Mrs. Peacock' : [19,1],'Professor Plum' : [6,1],'Miss Scarlet' : [1,17],'Colonel Mustard' : [8,24],'Mrs. White' : [25,15]}, main_player='Mr. Green', card_deck={}, confidential=[], hands={}, last_room={'Mr. Green' : 'none','Mrs. Peacock' : 'none','Professor Plum' : 'none','Miss Scarlet' : 'none','Colonel Mustard' : 'none','Mrs. White' : 'none'}):
        """Define variables to store the location of all characters/other variables. Called internally only.

        Ititializes a new instance of the board with all variables and players. If simulating the future game, then pass current pos and set simulation to True

        clue_game = setup_board()

        player_location(self,character),player_location_tile(self,character),return_player_locations_on_board(self),dice_roll(self),card_setup(self),navigate(self,can_i_move_here,character="Mr. Green"),visualize_map_using_return_player_locations_on_board(self),guessing(self),accuse(self),win_loss_check(self),player_memory(self),update_statistics(self),how_statistics(self)
        """
        self.current_pos = current_pos
        self.hands = hands
        self.card_deck = card_deck
        self.confidential = confidential
        self.main_player = main_player
        
        self.board_nav = [
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'ds',  1,   1, 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x',   1,   1,   1,   1,   1,   2,   1,   2,'dh', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x',   1,   1,   1,   1,   1,   1,   1,   1,   1, 'x', 'x', 'x', 'x', 'x', 'x',   1,   1,'dlo','x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1,   1, 'x', 'x', 'dh','dh','x', 'x',   1,   1,   2,   1,   1,   1,   1,   1, 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1,   1,   1,   2,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x','dli',  2,   1, 'x', 'x', 'x', 'x', 'x',   1,   1,   1,   2,   1,   1,   1,   1,   1, 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x',   1,   1, 'x','dd', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x','dli','x', 'x',   1,   1,   1, 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x',   2,   1,   2,   1,   1,   1,   1,   1, 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x','dbi','x', 'x', 'x', 'x',   1,   1,   1, 'x', 'x', 'x', 'x', 'x',   1,   2,'dd', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1,   1, 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1,   1, 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x','dbi',  2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1,   1,   2,   1,   1,   1,   1,   2,   1,   1,   1,   1,   1,   1,   1,   1, 'x', 'x'],
        ['x', 'x',   1,   1,   1,   1,   1,   1,   1, 'x','dba','x', 'x', 'x', 'x','dba','x',   1,   1,   1,   2,   1,   1,   1,   1, 'x'],
        ['x',   1,   1,   1,   1,   1,   1,   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x','dk', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x','dc',   2,   1,   2,'dba','x', 'x', 'x', 'x', 'x', 'x','dba',  2,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1,   1,   1, 'x', 'x', 'x', 'x',   1,   1,   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x',   1, 'x', 'x', 'x', 'x',   1, 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
        self.moves_left = 0
        self.room_list = ['dh', 'ds', 'dli', 'dbi', 'dba', 'dc', 'dk', 'dd', 'dlo']
        self.last_room = last_room
        self.goal = {name: ['x'] for name in copy.deepcopy(self.current_pos)}
        self.board_heatmap = [[0 for i in range(26)] for i in range(27)]
        self.search_board = {name:'x' for name in copy.deepcopy(self.current_pos)}
        self.q_hist = {name:'x' for name in copy.deepcopy(self.current_pos)}
        self.banned_rooms = {name: [] for name in copy.deepcopy(self.current_pos)}
        room = ['ds','dh','dlo','dd','dk','dba','dc','dbi','dli']
        self.hist = {'Mr. Green' : [[25,10]],'Mrs. Peacock' : [[19,1]],'Professor Plum' : [[6,1]],'Miss Scarlet' : [[1,17]],'Colonel Mustard' : [[8,24]],'Mrs. White' : [[25,15]]}
    def player_location(self,character='Mr. Green'):
        """return the position[x,y] for the current player location"""
        return self.current_pos[character]
    def player_location_tile(self,character='Mr. Green'):
        """Input character name ('Mr. Green', 'Mrs. Peacock','Professor Plum','Miss Scarlet','Colonel Mustard','Mrs. White')

        return the letter/number for the current player location"""
        
        return self.board_nav[self.current_pos[character][0]][self.current_pos[character][1]]
    def return_player_locations_on_board(self):
        
        board_deep_copy = []
        for i in self.board_nav:
            board_deep_copy= board_deep_copy + i
        for player_name in self.current_pos: #player tile = display player name
            board_deep_copy[self.player_location(player_name)[0]][self.player_location(player_name)[1]] = player_name.split(" ")[0][0] + player_name.split(" ")[1][0]
        return board_deep_copy
    def card_setup(self):
        """who = ['Miss Scarlet', 'Colonel Mustard', 'Mrs. White',  'Mr. Green',  'Mrs. Peacock', 'Professor Plum'],

        where = ['Study', 'Hall', 'Lounge', 'Dining Room', 'Kitchen', 'Ballroom', 'Conservatory', 'Billiard Room', 'Library'],

        weapon = ['Rope', 'Lead pipe', 'Knife', 'Wrench', 'Candlestick', 'Pistol']

        Returns random hands and confidential: hands = dict(player names), confidential = list [who,where,weapon]"""
        who = ['Miss Scarlet', 'Colonel Mustard', 'Mrs. White',  'Mr. Green',  'Mrs. Peacock', 'Professor Plum']
        where = ['Study', 'Hall', 'Lounge', 'Dining Room', 'Kitchen', 'Ballroom', 'Conservatory', 'Billiard Room', 'Library']
        weapon = ['Rope', 'Lead pipe', 'Knife', 'Wrench', 'Candlestick', 'Pistol']

        confidential = [random.choice(who), random.choice(where), random.choice(weapon)]
        who.remove(confidential[0])
        where.remove(confidential[1])
        weapon.remove(confidential[2])
        
        card_deck = who+where+weapon

        hands = {
            'Miss Scarlet' : [],
            'Colonel Mustard' : [],
            'Mrs. White' : [],
            'Mr. Green' : [],
            'Mrs. Peacock' : [],
            'Professor Plum' : []
            }

        random.shuffle(card_deck)
        while len(card_deck) > 0:
            for i in hands:
                hands[i].append(card_deck[0])
                card_deck.remove(card_deck[0])

        self.hands = hands
        self.card_deck = card_deck
        self.confidential = confidential
        return hands, card_deck, confidential
    def blocking_locations(self):
        
        player_blocking_locations = []
        for player_coordinates in self.current_pos:
            player_blocking_locations = player_blocking_locations + [self.current_pos[player_coordinates]]
        return player_blocking_locations
    def navigate(self,can_i_move_here,character="Mr. Green"):
        
        """navigate(can_i_move_here,character="Mr. Green")
        
        can_i_move_here accepts values between 1-4 for hallways. Input 5 to enter room 
        
        character determines what character is moved"""
        pos_val = self.player_location_tile(character) 
        player_blocking_locations = self.blocking_locations()

        if pos_val not in self.room_list:         #if not currently in a room and not bloacked by player
            p1 = [self.player_location(character)[0]-1, self.player_location(character)[1]]
            p2 = [self.player_location(character)[0]+1, self.player_location(character)[1]]
            p3 = [self.player_location(character)[0], self.player_location(character)[1]-1]
            p4 = [self.player_location(character)[0],self.player_location(character)[1]+1]
            move = 'Invalid location'
            if can_i_move_here == 5:
                for i in [p1,p2,p3,p4]:
                    if i not in [player_blocking_locations] and self.board_nav[i[0]][i[1]] not in [pos_val, 1, 2, 'x']:
                        potential_pos = i
                        move = self.board_nav[potential_pos[0]][potential_pos[1]]
            elif can_i_move_here == 1 and p1 not in player_blocking_locations:#up 1?
                potential_pos = p1
                move = self.board_nav[potential_pos[0]][potential_pos[1]]
            elif can_i_move_here == 2 and p2 not in player_blocking_locations:# down 2?
                potential_pos = p2
                move = self.board_nav[potential_pos[0]][potential_pos[1]]
            elif can_i_move_here == 3 and p3 not in player_blocking_locations:#left 3?
                potential_pos = p3
                move = self.board_nav[potential_pos[0]][potential_pos[1]]
            elif can_i_move_here == 4 and p4 not in player_blocking_locations:#right 4?
                potential_pos = p4
                move = self.board_nav[potential_pos[0]][potential_pos[1]]
            else: return self.current_pos[character]#terminates with no change if movement to the location is not possible
            #print('Wants to move to: ', move)
            
            
            #if not occupied -- best way to keep track of occupied?
            if move in [1,2] and potential_pos not in self.hist[character]: #and not occupied:
                self.current_pos[character] = potential_pos
                pos_val = move
                self.hist[character]=self.hist[character]+[self.current_pos[character]]
                self.moves_left = self.moves_left - 1
            elif move not in ['x',self.last_room[character]] and potential_pos not in self.hist[character]:   #if the move is through a doorway to get into a room
                if pos_val == 2:
                    self.current_pos[character] = potential_pos
                    pos_val = move
                    self.moves_left = 0     #turn is over when it enters a room
                    self.last_room[character] = move
        else:    #if in a room
            if pos_val == 'ds' or pos_val == 'dlo' or pos_val == 'dk' or pos_val == 'dc':
                if can_i_move_here == 1:
                    if pos_val == 'ds': self.current_pos[character] = [19,20]
                    elif pos_val == 'dk': self.current_pos[character] = [4,7]
                    elif pos_val == 'dc': self.current_pos[character] = [6,18]
                    elif pos_val == 'dlo': self.current_pos[character] = [20,5]
                    self.last_room[character] = pos_val
                    self.moves_left = 0         #turn is over when it uses a secret passageway
                elif can_i_move_here == 2:
                    if pos_val == 'ds': self.current_pos[character] = [5,7]
                    elif pos_val == 'dk': self.current_pos[character] = [18,20]
                    elif pos_val == 'dc': self.current_pos[character] = [20,6]
                    elif pos_val == 'dlo': self.current_pos[character] = [7,18]
                    self.moves_left = self.moves_left -1
            elif pos_val == 'dh':
                if can_i_move_here == 1: self.current_pos[character] = [5,9]
                elif can_i_move_here == 2: self.current_pos[character] = [8,12]
                elif can_i_move_here == 3: self.current_pos[character] = [8,13]
                self.moves_left = self.moves_left -1
            elif pos_val == 'dd':
                if can_i_move_here == 1: self.current_pos[character] = [9,18]
                elif can_i_move_here == 2: self.current_pos[character] = [13,16]
                self.moves_left = self.moves_left -1
            elif pos_val == 'dba':
                if can_i_move_here == 1: self.current_pos[character] = [20,8]
                elif can_i_move_here == 2: self.current_pos[character] = [17,10]
                elif can_i_move_here == 3: self.current_pos[character] = [17,15]
                elif can_i_move_here == 4: self.current_pos[character] = [20,17]
                self.moves_left = self.moves_left -1
            elif pos_val == 'dbi':
                if can_i_move_here == 1: self.current_pos[character] = [16,7]
                elif can_i_move_here == 2: self.current_pos[character] = [12,2]
                self.moves_left = self.moves_left -1
            elif pos_val == 'dli':
                if can_i_move_here == 1: self.current_pos[character] = [9,8]
                elif can_i_move_here == 2: self.current_pos[character] = [12,4]
                self.moves_left = self.moves_left -1
        return self.current_pos[character]
    def manhatten_distance(self,coordinates,list_of_possible_coordinates):
        min_dist = 10000000
        x,y = coordinates
        move = []
        for coor in list_of_possible_coordinates:
            diff_x= abs(coor[0] -x)
            diff_y= abs(coor[1] -y)
            diff = diff_x + diff_y
            if min_dist > diff:
                min_dist = diff
                move = [coor[0], coor[1]]
        return move,min_dist
    def euclidean_distance(self,coordinates,list_of_possible_coordinates):
        min_dist = 10000000
        x,y = coordinates
        #else: 
            #print(coordinates)
            #sleep(29856)
        move = []
        if len(list_of_possible_coordinates) == 1: return list_of_possible_coordinates[0]
        for coor in list_of_possible_coordinates:
            diff_x= abs(coor[0] -x)
            diff_y= abs(coor[1] -y)
            diff = (diff_x**2 + diff_y**2)**1/2
            if min_dist > diff:
                min_dist = diff
                move = [coor[0], coor[1]]
                
        #add simulated annealing to shake infinte loops 
        #if random.random() < .5:
           # move = random.choice(list_of_possible_coordinates)
        return move   
    def closest_auto_move(self,character="Mr. Green",case_file_rooms=None,player_type='logic',rl_input=[],possible_moves=[]):
        """Moves the named bot to the closest door we need prob info on, and then enters."""
        if player_type == 'logic' and OLD_LOGIC:
            player_blocking_locations = self.blocking_locations()
            room_dict = {'dh':'Hall', 'ds':'Study', 'dli':'Library', 'dbi':'Billiard Room', 'dba':'Ballroom', 'dc':'Conservatory','dk':'Kitchen', 'dd':'Dining Room', 'dlo':'Lounge','none':''}
            dict_of_doors_to_be_pruned ={'Study' : [[5, 7]], 'Hall' : [[5, 9], [8, 12], [8, 13]], 'Lounge' : [[7, 18]], 'Dining Room' : [[9, 18], [13, 16]], 'Kitchen' : [[18, 20]], 'Ballroom' : [[20, 8], [20, 17], [17, 10], [17, 15]], 'Conservatory' : [[20, 6]], 'Billiard Room' : [[16, 7], [12, 2]], 'Library' : [[9, 8], [12, 4]]}
            doors = []
            
            for set_of_rooms in case_file_rooms:
                if set_of_rooms != room_dict[self.last_room[character]]:
                    for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                        if room not in player_blocking_locations:
                            doors.append(room)
            if len(doors) == 0:
                for set_of_rooms in dict_of_doors_to_be_pruned:
                    if set_of_rooms != room_dict[self.last_room[character]]:
                        for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                            if room not in player_blocking_locations:
                                doors.append(room)
            hist = []
            role = self.moves_left
            closest_door = self.euclidean_distance(self.current_pos[character],doors)#does not account for trap doors
            for moves in range(40):
                #for moves in range(40):#cap max number of move attempts
                x,y = self.current_pos[character]
                closest_door = self.euclidean_distance([x,y],doors)

                possible_moves = []#gets 4 possible legal moves
                for i in [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]:
                    if self.board_nav[i[0]][i[1]] not in ['x'] + [self.last_room[character]] and i not in hist  + player_blocking_locations:
                        possible_moves.append(i)
                        hist.append(i)
                    elif self.board_nav[i[0]][i[1]] not in ['x'] + [self.last_room[character]] and i not in player_blocking_locations and len(possible_moves)<1:
                        possible_moves.append(i)
                        hist.append(i)

                if len(possible_moves) != 0:
                    closest_step = self.euclidean_distance(closest_door,possible_moves)
                    
                    if  [x-1,y]   == closest_step: move = 1
                    elif [x+1,y]  == closest_step: move = 2
                    elif [x,y-1]  == closest_step: move = 3
                    elif [x,y+1]  == closest_step: move = 4
                    self.navigate(move,character)
                    self.moves_left = role - len(self.hist[character])
                    if self.current_pos[character] in doors:
                        self.navigate(5, character)
                        break
                    elif  self.moves_left < 1: break
                else:
                    break

        elif player_type == 'logic':
            player_blocking_locations = copy.deepcopy(self.blocking_locations())#self.blocking_locations() + self.banned_rooms[character]
            
            room_dict = {'dh':'Hall', 'ds':'Study', 'dli':'Library', 'dbi':'Billiard Room', 'dba':'Ballroom', 'dc':'Conservatory','dk':'Kitchen', 'dd':'Dining Room', 'dlo':'Lounge','none':'','x':'x'}
            dict_of_doors_to_be_pruned ={'Study' : [[5, 7]], 'Hall' : [[5, 9], [8, 12], [8, 13]], 'Lounge' : [[7, 18]], 'Dining Room' : [[9, 18], [13, 16]], 'Kitchen' : [[18, 20]], 'Ballroom' : [[20, 8], [20, 17], [17, 10], [17, 15]], 'Conservatory' : [[20, 6]], 'Billiard Room' : [[16, 7], [12, 2]], 'Library' : [[9, 8], [12, 4]]}
            doors = []#[dict_of_doors_to_be_pruned[list][0] for list in dict_of_doors_to_be_pruned]
            
            for set_of_rooms in case_file_rooms:
                if set_of_rooms != room_dict[self.last_room[character]]:
                    for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                        if room not in player_blocking_locations:
                            doors.append(room)
            if len(doors) == 0:
                for set_of_rooms in dict_of_doors_to_be_pruned:
                    if set_of_rooms != room_dict[self.last_room[character]]:
                        for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                            if room not in player_blocking_locations:
                                doors.append(room)
            
            #cost spent and cost to get there adjacent BFS
            if self.goal[character] ==  ['x']:
                self.goal[character] = copy.deepcopy(random.choice(doors))
                
            role = self.moves_left
            hist = [self.current_pos[character]]
            x,y = copy.deepcopy(self.current_pos[character][0]),copy.deepcopy(self.current_pos[character][1])
            
            if self.q_hist[character] == 'x':
                self.q_hist[character] = []
            if self.search_board[character] == 'x':
                self.search_board[character] = copy.deepcopy(self.board_nav)
                
            def expand_node(self,x,y,search_board,q_hist):
                for i in [[x,y],[x,y-1],[x,y+1],[x-1,y],[x+1,y]]:
                    if self.board_nav[i[0]][i[1]] not in room_dict:
                        search_board[i[0]][i[1]] = self.manhatten_distance(i,[self.goal[character]])[1]+self.manhatten_distance(i,[self.current_pos[character]])[1]
                        for f in q_hist:
                            if f == i:
                                search_board[i[0]][i[1]] = search_board[i[0]][i[1]] + 1
                                
                return search_board,q_hist
            
            def expand_map(self,x,y,search_board,q_hist):
                for x in range(27):
                    for y in range(26):
                        i = [x,y]
                        if self.board_nav[x][y] not in ['x']:
                            search_board[x][y] = self.manhatten_distance(i,[self.goal[character]])[1]+self.manhatten_distance(i,[self.current_pos[character]])[1]
                            for f in q_hist:
                                if f == i:
                                    search_board[x][y] = search_board[x][y] + 1
                                
                return search_board,q_hist
            
            def move_lowest(self,x,y,search_board,q_hist):
                lowest = 10000
                lowest_i = [x,y]
                for i in [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]:
                    if type(search_board[i[0]][i[1]]) is not str:
                        if lowest > search_board[i[0]][i[1]]:
                            lowest = search_board[i[0]][i[1]]
                            lowest_i = i
                            q_hist.append(i)
                return lowest_i[0],lowest_i[1],q_hist
            
            # for _ in range(100):
            #     search_board,q_hist = expand_node(self,x,y,search_board,q_hist)
            #     x,y,q_hist = move_lowest(self,x,y,search_board,q_hist)
            #search_board,q_hist = expand_map(self,x,y,search_board,q_hist)
            while [x,y] != self.goal[character]:
                 self.search_board[character],self.q_hist[character] = expand_map(self,x,y,self.search_board[character],self.q_hist[character])
                 x,y,self.q_hist[character] = move_lowest(self,x,y,self.search_board[character],self.q_hist[character])
            
            
            for j in self.q_hist[character]:
                if self.q_hist[character].count(j) > 1:
                    index_first = self.q_hist[character].index(j)
                    for ind in range(index_first+1,len(self.q_hist[character])):
                        if self.q_hist[character][ind]==j:
                            second_index = ind
                    a = len(self.q_hist[character])
                    self.q_hist[character] = self.q_hist[character][0:index_first]+self.q_hist[character][second_index:-1]
                    b = len(self.q_hist[character])
                    print(a-b,end="\r")
            print(self.q_hist[character])
            
            count = 0
            while len(self.q_hist[character])>0 and count < 40:
                count+=1
                x,y = self.current_pos[character][0],self.current_pos[character][1]
                move = self.q_hist[character][0]#random.choice([i for i in [[x,y-1],[x,y+1],[x-1,y],[x+1,y]] if i not in hist])
                del self.q_hist[character][0]
                hist.append(move)
                if  [x-1,y]   == move: move = 1
                elif [x+1,y]  == move: move = 2
                elif [x,y-1]  == move: move = 3
                elif [x,y+1]  == move: move = 4
                # else: 
                #     self.current_pos[character] = move
                #     self.hist[character].append(move)
                #     self.moves_left = role - len(self.hist[character])
                
                old = copy.deepcopy(self.current_pos[character])
                if move in [1,2,3,4,5]:
                    self.navigate(move,character)
                    self.moves_left = role - len(self.hist[character])
                x,y = self.current_pos[character][0],self.current_pos[character][1]
                
                if self.goal[character] == [x,y]:
                    self.goal[character] = ['x']
                    self.q_hist[character] = 'x'
                    self.search_board[character] = 'x'
                    
                if [x,y] in doors:
                    self.navigate(5, character)
                    break
                    if old != self.current_pos[character]: break
                    
                if self.goal[character] == ['x']:
                    break
        
        elif player_type == 'RL':
            global model
            player_blocking_locations = self.blocking_locations()
            doors = copy.deepcopy(possible_moves)
            hist = []
            role = self.moves_left
            #Selects four closest doors based on euclidean distance and then runs the RL to determine which should be entered
            four_closest_doors = []
            if HYBRID_EUCLIDEAN_PEACOCK:
                while len(four_closest_doors) < 4 and len(doors) > 0:
                    closest_door = self.manhatten_distance(self.current_pos[character][0],doors)#does not account for trap doors
                    #print(closest_door)
                    four_closest_doors.append(closest_door)
                    doors.remove(closest_door)
                    
            else: four_closest_doors = copy.deepcopy(doors)
            
            highest_value_doors = model.predict(rl_input).tolist()
            index_list = []
            for i in four_closest_doors:
                index_list.append(highest_value_doors[possible_moves.index(i)])
            max_value = possible_moves[highest_value_doors.index(max([i for i in highest_value_doors if i in index_list]))]
            ######enter room/go towards
            for moves in range(40):
                x,y = self.current_pos[character]
                possible_moves = []#gets 4 possible legal moves
                closest_door = copy.deepcopy(max_value)#self.manhatten_distance([x,y],doors)[0]
                
                for i in [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]:
                    if self.board_nav[i[0]][i[1]] not in ['x'] + [self.last_room[character]] and i not in hist  + player_blocking_locations:
                        possible_moves.append(i)
                        hist.append(i)
                    elif self.board_nav[i[0]][i[1]] not in ['x'] + [self.last_room[character]] and i not in player_blocking_locations and len(possible_moves)<1:
                        possible_moves.append(i)
                        hist.append(i)

                if len(possible_moves) != 0:
                    #closest step to selected door
                    closest_step = self.manhatten_distance(closest_door,possible_moves)[0]
                    
                    if  [x-1,y]   == closest_step: move = 1
                    elif [x+1,y]  == closest_step: move = 2
                    elif [x,y-1]  == closest_step: move = 3
                    elif [x,y+1]  == closest_step: move = 4
                    self.navigate(move,character)
                    self.moves_left = role - len(self.hist[character])
                    if self.current_pos[character] in doors:
                        self.navigate(5, character)                
    def reset_hist(self,character="Mr. Green"):
        self.hist[character] = [self.current_pos[character]]
class metrics_and_tables():
    def __init__(self,all_cards,player_thresholds={'Miss Scarlet':1, 'Colonel Mustard':1, 'Mrs. White':1,  'Mr. Green':1,  'Mrs. Peacock':1, 'Professor Plum':1}):
        self.player_win_times = {}
        for name in ['Miss Scarlet', 'Colonel Mustard', 'Mrs. White',  'Mr. Green',  'Mrs. Peacock', 'Professor Plum']:
            self.player_win_times[name] = 0
        self.total_games = 0
        self.average_turns = 60
        self.neurons = 0
        self.all_cards = all_cards
        self.predict_dataset = None
        self.predict_reward_labels = []
        self.predict_names_lineup = []
        self.predict_collected_input = []
        self.collected_input = []
        self.predict_labels = []
        self.average_threshold = .75
        self.history = None
        self.labels =[]
        self.reward_labels = []
        self.monte_carlo_heuristic = None
        self.dataset = None
        self.testds = None
        self.names_lineup = []
        self.outcome = 'record turn'
        self.board_heatmap = [[0 for i in range(26)] for i in range(27)]
        self.player_thresholds = player_thresholds
    def reset(self):
        self.player_win_times = {}
        for name in ['Miss Scarlet', 'Colonel Mustard', 'Mrs. White',  'Mr. Green',  'Mrs. Peacock', 'Professor Plum']:
            self.player_win_times[name] = 0
        self.total_games = 0
        self.board_heatmap = [[0 for i in range(26)] for i in range(27)]
        #self.threshold_q = np.zeros((state_size, action_size))
        #self.guess_q = np.zeros((state_size, action_size))
        #self.answer_deception_q = np.zeros((state_size, action_size))
        #self.move_q = np.zeros((state_size, action_size))#state size actions size
    def win_rate(self,character):
        if self.total_games == 0:
            return 0
        return self.player_win_times[character]/self.total_games
    def update_best_threshold_greedy(self):
        max = 0
        max_name=""
        for name in self.player_win_times:
            if max < self.player_win_times[name]:
                max = self.player_win_times[name]
                max_name = name
        for name in self.player_thresholds:
            if name != max_name:
                self.player_thresholds[name] = (random.random()+(4*self.player_thresholds[max_name]))/5
        return f"Threshold {max_name},{self.player_thresholds[max_name]}"
    def process_current_data(self,turns=0,board=None,player_cards=None,confidential=None,player_locations=None,character=None,stored_player_guesses=None,final_player_certainty=None,reward=0):
        if self.outcome == 'record turn':#convert all relevent player info into a single tensor input. Specify who's turn it is on a seperate file
            cards_to_nums = {'Miss Scarlet':1, 'Colonel Mustard':2, 'Mrs. White':3,  'Mr. Green':4,  'Mrs. Peacock':5, 'Professor Plum':6,'Study':1, 'Hall':2, 'Lounge':3, 'Dining Room':4, 'Kitchen':5, 'Ballroom':6, 'Conservatory':7, 'Billiard Room':8, 'Library':9,'Rope':1, 'Lead pipe':2, 'Knife':3, 'Wrench':4, 'Candlestick':5, 'Pistol':6}
            inp1 = []
            inp1.append(cards_to_nums[character])  
            for name in player_locations:#locations and board heatmap
                x,y = player_locations[name][0],player_locations[name][1]
                self.board_heatmap[x][y] += 1
                
                for x_y in player_locations[name]:
                    inp1.append(x_y)        
            for x in range(27):
                for x_y in self.board_heatmap[x]:
                    inp1.append(x_y)    
            for name in player_cards:
                for all_card in self.all_cards:#player possible cards
                    if all_card in player_cards[name]:
                        inp1.append(1)
                    else:
                        inp1.append(0)  
                    if all_card in stored_player_guesses[name]:
                        inp1.append(1)
                    else:
                        inp1.append(0) 
                    if all_card in confidential:#confidential
                        inp1.append(1)
                    else:
                        inp1.append(0)  
            inp1.append(turns)      
            self.neurons = len(inp1)
            self.collected_input.append(tf.convert_to_tensor(inp1,dtype=dtype))
            inp1 = []
            self.names_lineup.append(character)
            #self.reward_labels.append(tf.convert_to_tensor([final_player_certainty],dtype=dtype))      
        elif self.outcome in ['Miss Scarlet', 'Colonel Mustard', 'Mrs. White',  'Mr. Green',  'Mrs. Peacock', 'Professor Plum']:#Line up labels of who won
            #set labels for loser and winnder for all moves in the past game
            len_collection = len(self.collected_input)
            old_len_labels = len(self.labels)
            for _ in range(len_collection - old_len_labels):
                self.labels.append([100])
            for list1 in range(old_len_labels,len_collection):#optimized to not slow down for large lists
                if self.labels[list1]==[100]:
                    if self.names_lineup[list1] == self.outcome:
                        self.labels[list1] = tf.convert_to_tensor([1],dtype=dtype)
                    else: 
                        self.labels[list1] = tf.convert_to_tensor([0],dtype=dtype)
                        
            #set labels for the reward function, aka how much infomration the bot will know in the future
            #only works with single players
            old_reward_labels = len(self.reward_labels)
            for _ in range(len_collection - old_reward_labels):
                self.reward_labels.append(tf.convert_to_tensor([reward],dtype=dtype))
            #print(final_player_certainty)
                    
            self.board_heatmap = [[0 for i in range(26)] for i in range(27)]
        elif self.outcome == 'Conv to dataset':#Store only in mem
            #self.dataset = tf.data.Dataset.from_tensor_slices((self.collected_input,self.labels))
            if self.testds == None:
                self.testds = tf.data.Dataset.from_tensor_slices((self.collected_input,self.reward_labels))
                self.names_lineup,self.collected_input,self.labels,self.reward_labels = [],[],[],[]
                self.testds = self.testds.batch(50, drop_remainder=False)     
            else:
                self.dataset = tf.data.Dataset.from_tensor_slices((self.collected_input,self.reward_labels))
                self.names_lineup,self.collected_input,self.labels,self.reward_labels = [],[],[],[]
                self.dataset = self.dataset.batch(50, drop_remainder=False)    
        elif self.outcome == 'RL':
            cards_to_nums = {'Miss Scarlet':1, 'Colonel Mustard':2, 'Mrs. White':3,  'Mr. Green':4,  'Mrs. Peacock':5, 'Professor Plum':6,'Study':1, 'Hall':2, 'Lounge':3, 'Dining Room':4, 'Kitchen':5, 'Ballroom':6, 'Conservatory':7, 'Billiard Room':8, 'Library':9,'Rope':1, 'Lead pipe':2, 'Knife':3, 'Wrench':4, 'Candlestick':5, 'Pistol':6}
            inp1 = []
            inp1.append(cards_to_nums[character])  
            for name in player_locations:#locations and board heatmap
                x,y = player_locations[name][0],player_locations[name][1]
                #self.board_heatmap[x][y] += 1
                
                for x_y in player_locations[name]:
                    inp1.append(x_y)        
            for x in range(27):
                for x_y in self.board_heatmap[x]:
                    inp1.append(x_y)    
            for name in player_cards:
                for all_card in self.all_cards:#player possible cards
                    if all_card in player_cards[name]:
                        inp1.append(1)
                    else:
                        inp1.append(0)  
                    if all_card in stored_player_guesses[name]:
                        inp1.append(1)
                    else:
                        inp1.append(0) 
                    if all_card in confidential:#confidential
                        inp1.append(1)
                    else:
                        inp1.append(0)  
            inp1.append(turns)      

            self.predict_collected_input.append(tf.convert_to_tensor(inp1,dtype=dtype))
            inp1 = []
            self.predict_names_lineup.append(character)     
        elif self.outcome == 'RL dataset':     
            self.predict_dataset = tf.data.Dataset.from_tensor_slices((self.predict_collected_input))
            self.predict_names_lineup,self.predict_collected_input,self.predict_labels,self.predict_reward_labels = [],[],[],[]
            self.predict_dataset = self.predict_dataset.batch(1, drop_remainder=False)               
    def model(self,activation='relu',loss='binary_crossentropy'):
        current_path = os.getcwd()
        if self.monte_carlo_heuristic == None:
            input = Input(shape=(self.neurons,))
            x1 = Dense(self.neurons,activation=activation)(input)
            x2 = Dense(self.neurons/(3/4),activation=activation)(x1)
            x3 = Dense(self.neurons/(2/4),activation=activation)(x2)
            x4 = Dense(self.neurons/(1/4),activation=activation)(x3)
            output = Dense(1,activation=None,dtype='float32')(x4)
            self.monte_carlo_heuristic  = Model(input,output)
            self.monte_carlo_heuristic.compile(loss=loss,optimizer='adam',metrics=[loss])
            self.monte_carlo_heuristic.summary()
        
        self.history = self.monte_carlo_heuristic.fit(self.dataset,validation_data=self.testds,epochs=100,shuffle=True,callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=3))#,validation_split=0,,class_weight={0:1,1:6}
#BRIAN GUI INSERTION###########################################       
# declare classes for player widgets
class GamePiece1(Widget):
    pos_x = NumericProperty(0)
    pos_y = NumericProperty(0)
    new_pos = ReferenceListProperty(pos_x, pos_y)

    def move_piece(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.pos = Vector(*self.new_pos)
class GamePiece2(Widget):
    pos_x = NumericProperty(0)
    pos_y = NumericProperty(0)
    new_pos = ReferenceListProperty(pos_x, pos_y)

    def move_piece(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.pos = Vector(*self.new_pos)
class GamePiece3(Widget):
    pos_x = NumericProperty(0)
    pos_y = NumericProperty(0)
    new_pos = ReferenceListProperty(pos_x, pos_y)

    def move_piece(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.pos = Vector(*self.new_pos)
class GamePiece4(Widget):
    pos_x = NumericProperty(0)
    pos_y = NumericProperty(0)
    new_pos = ReferenceListProperty(pos_x, pos_y)
    def move_piece(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.pos = Vector(*self.new_pos)
class GamePiece5(Widget):
    pos_x = NumericProperty(0)
    pos_y = NumericProperty(0)
    new_pos = ReferenceListProperty(pos_x, pos_y)

    def move_piece(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.pos = Vector(*self.new_pos)
class GamePiece6(Widget):
    pos_x = NumericProperty(0)
    pos_y = NumericProperty(0)
    new_pos = ReferenceListProperty(pos_x, pos_y)
    def move_piece(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.pos = Vector(*self.new_pos)
# define main widget class and functions..
class MainWidget(FloatLayout):
    players = ['Miss Scarlet', "Colonel Mustard", "Mrs. White", "Mr. Green", "Mrs. Peacock", "Professor Plum"]
    coord_list = [[1, 17], [8, 24], [25, 15], [25, 10], [19, 1], [6, 1]]
    player = ''
    piece_red = ObjectProperty(None)
    piece_yellow = ObjectProperty(None)
    piece_white = ObjectProperty(None)
    piece_green = ObjectProperty(None)
    piece_blue = ObjectProperty(None)
    piece_purple = ObjectProperty(None)

    who = ['Miss Scarlet', 'Colonel Mustard', 'Mrs. White',  'Mr. Green',  'Mrs. Peacock', 'Professor Plum']
    where = ['Study', 'Hall', 'Lounge', 'Dining Room', 'Kitchen', 'Ballroom', 'Conservatory', 'Billiard Room', 'Library']
    weapon = ['Rope', 'Lead pipe', 'Knife', 'Wrench', 'Candlestick', 'Pistol']
    all_cards = who + where + weapon
    metrics = metrics_and_tables(all_cards)
    metrics.outcome = 'record turn'

    room_dict = {'dh':'Hall', 'ds':'Study', 'dli':'Library', 'dbi':'Billiard Room', 'dba':'Ballroom', 'dc':'Conservatory','dk':'Kitchen', 'dd':'Dining Room', 'dlo':'Lounge'}
    turns = 0
    player_num = 6
    # initialize/reset board, players, hands, probabilities
    game_over = False
    clue_game = setup_board()
    clue_game.card_setup()
    if HUMAN_PLAYER:
        player1 = Player(name="Miss Scarlet", current_pos=[1, 17], hand=clue_game.hands["Miss Scarlet"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Miss Scarlet"],all_cards=all_cards,player_type='human')
    else: 
        player1 = Player(name="Miss Scarlet", current_pos=[1, 17], hand=clue_game.hands["Miss Scarlet"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Miss Scarlet"],all_cards=all_cards,player_type='logic')
    player2 = Player(name="Colonel Mustard", current_pos=[8, 24], hand=clue_game.hands["Colonel Mustard"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Colonel Mustard"],all_cards=all_cards)
    player3 = Player(name="Mrs. White", current_pos=[25, 15], hand=clue_game.hands["Mrs. White"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mrs. White"],all_cards=all_cards)
    player4 = Player(name="Mr. Green", current_pos=[25, 10], hand=clue_game.hands["Mr. Green"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mr. Green"],all_cards=all_cards)
    if RL_AGENT_AS_MRS_PEACOCK:
        player5 = Player(name="Mrs. Peacock", current_pos=[19, 1], hand=clue_game.hands["Mrs. Peacock"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mrs. Peacock"],all_cards=all_cards,player_type='RL')
    else: 
        player5 = Player(name="Mrs. Peacock", current_pos=[19, 1], hand=clue_game.hands["Mrs. Peacock"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mrs. Peacock"],all_cards=all_cards,player_type='logic')
    player6 = Player(name="Professor Plum", current_pos=[6, 1], hand=clue_game.hands["Professor Plum"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Professor Plum"],all_cards=all_cards)

    # using a queue data struct - we can "pop" the first index and then append it to cycle through the list of players and maintain order
    player_order = deque([player1, player2, player3, player4, player5, player6])
    current_player = player_order.popleft()

    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        Clock.schedule_once(self.init_player_positions, 0)
        self.turns = 0
    def on_touch_down(self, touch):
        if not self.game_over:
            Clock.schedule_once(self.play_game, 0)
            Clock.schedule_once(self.init_player_positions, 0)
        else:
            self.turns = 0
            print(f"Game over! {self.current_player.name} correctly guessed it was {self.clue_game.confidential[0]} in the {self.clue_game.confidential[1]} with the {self.clue_game.confidential[2]}")
            exit()
    def generate_new_coordinates(self):
        # Written by Nayana - updated by Harshitha and Brian, used for initial GUI testing
        new_coord_list = []
        for _ in range(6):
            x = random.randint(1, 24)
            y = random.randint(1, 25)
            new_coord_list.append([x, y])
        return new_coord_list
    def init_player_positions(self, dt):
        for i in range(0, len(self.players)):
            self.player = self.players[i]
            if self.player == "Miss Scarlet":
                x, y = self.get_coords('Miss Scarlet')
                self.piece_red.move_piece(self.width*x/26, self.height*y/27)
            if self.player == "Colonel Mustard":
                x, y = self.get_coords('Colonel Mustard')
                self.piece_yellow.move_piece(self.width*x/26, self.height*y/27)
            if self.player == "Mrs. White":
                x, y = self.get_coords('Mrs. White')
                self.piece_white.move_piece(self.width*x/26, self.height*y/27)
            if self.player == "Mr. Green":
                x, y = self.get_coords('Mr. Green')
                self.piece_green.move_piece(self.width*x/26, self.height*y/27)
            if self.player == "Mrs. Peacock":
                x, y = self.get_coords('Mrs. Peacock')
                self.piece_blue.move_piece(self.width*x/26, self.height*y/27)
            if self.player == "Professor Plum":
                x, y = self.get_coords('Professor Plum')
                self.piece_purple.move_piece(self.width*x/26, self.height*y/27)
    def get_coords(self, name):
        for i in range(0, len(self.players)):
            if self.players[i] == name:
                x = self.coord_list[i][1]
                y = 26 - self.coord_list[i][0]
        return x, y
    def update_coords(self):
        for i in range(0, len(self.players)):
            if self.current_player.name == self.players[i]:
                self.coord_list[i] = self.clue_game.current_pos[self.players[i]]
            for p in self.player_order:
                if p.name == self.players[i]:
                    self.coord_list[i] = self.clue_game.current_pos[self.players[i]]
    def play_game(self, dt):
        self.turns = self.turns + 1
        self.clue_game.reset_hist(self.current_player.name)
        if not self.current_player.loser:
            self.current_player.earned_rewards += int(self.current_player.earned_rewards*.94)
            #move
            self.clue_game.moves_left = self.current_player.dice_roll()
            moves = 0
            role = self.clue_game.moves_left
            if self.current_player.player_type == 'human':#human player
                while self.clue_game.moves_left > 0 and moves < 40:
                    #############################################################
                    possible_moves = []#gets 4 possible legal moves
                    x,y = self.clue_game.current_pos[self.current_player.name]
                    for i in [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]:
                        if self.clue_game.board_nav[i[0]][i[1]] not in ['x'] + [self.clue_game.last_room[self.current_player.name]] and i not in self.clue_game.blocking_locations():
                            possible_moves.append(i)
                    human_move = None
                    i = int(input(f"Input one of these: {possible_moves} = {[i for i in range(len(possible_moves))]} or 5 to enter room: "))
                    if i != 5:
                        human_move = possible_moves[i]
                    if  [x-1,y]   == human_move: human_move = 1
                    elif [x+1,y]  == human_move: human_move = 2
                    elif [x,y-1]  == human_move: human_move = 3
                    elif [x,y+1]  == human_move: human_move = 4
                    print(human_move)
                    self.clue_game.navigate(human_move,self.current_player.name)
                    if human_move == None:
                        print("Out of possible moves")
                    moves += 1
                    self.clue_game.moves_left = role - len(self.clue_game.hist[self.current_player.name])
            
            
            else:#none human player
                if self.current_player.player_type == 'RL':
                    self.current_player.check_ready_to_accuse()
                    possible_moves = []#gets 4 possible legal moves
                    x,y = self.clue_game.current_pos[self.current_player.name]
                    l = copy.deepcopy(self.clue_game.current_pos)
                    player_blocking_locations = self.clue_game.blocking_locations()
                    room_dict = {'dh':'Hall', 'ds':'Study', 'dli':'Library', 'dbi':'Billiard Room', 'dba':'Ballroom', 'dc':'Conservatory','dk':'Kitchen', 'dd':'Dining Room', 'dlo':'Lounge','none':''}
                    dict_of_doors_to_be_pruned ={'Study' : [[5, 7]], 'Hall' : [[5, 9], [8, 12], [8, 13]], 'Lounge' : [[7, 18]], 'Dining Room' : [[9, 18], [13, 16]], 'Kitchen' : [[18, 20]], 'Ballroom' : [[20, 8], [20, 17], [17, 10], [17, 15]], 'Conservatory' : [[20, 6]], 'Billiard Room' : [[16, 7], [12, 2]], 'Library' : [[9, 8], [12, 4]]}
                    doors = []
                    
                    for set_of_rooms in self.current_player.case_file_where_prob:
                        if set_of_rooms != room_dict[self.clue_game.last_room[self.current_player.name]]:
                            for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                                if room not in player_blocking_locations:
                                    doors.append(room)
                    if len(doors) == 0:
                        for set_of_rooms in dict_of_doors_to_be_pruned:
                            if set_of_rooms != room_dict[self.clue_game.last_room[self.current_player.name]]:
                                for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                                    if room not in player_blocking_locations:
                                        doors.append(room)
                    
                    self.metrics.outcome = 'RL'
                    for coordinates in doors:
                            possible_moves.append(coordinates)
                            l[self.current_player.name] = coordinates#alter the reported current location of the RL player for all options then pass into a dataset
                            self.metrics.process_current_data(turns=self.turns,board=self.clue_game.board_nav,player_cards=self.current_player.all_players_possible_cards,confidential=self.current_player.case_file_who_prob+self.current_player.case_file_where_prob+self.current_player.case_file_weapon_prob,player_locations=l,character=self.current_player.name,stored_player_guesses=self.current_player.stored_player_guesses,final_player_certainty=self.current_player.certainty,reward=self.current_player.earned_rewards)
                    self.metrics.outcome = 'RL dataset'#create dataset
                    self.metrics.process_current_data(turns=self.turns,board=self.clue_game.board_nav,player_cards=self.current_player.all_players_possible_cards,confidential=self.current_player.case_file_who_prob+self.current_player.case_file_where_prob+self.current_player.case_file_weapon_prob,player_locations=l,character=self.current_player.name,stored_player_guesses=self.current_player.stored_player_guesses,final_player_certainty=self.current_player.certainty,reward=self.current_player.earned_rewards)
                    #check for rooms
                    self.clue_game.closest_auto_move(self.current_player.name,self.current_player.case_file_where_prob,player_type='RL',rl_input=self.metrics.predict_dataset,possible_moves=possible_moves)
                    if TRAIN and self.current_player.player_type == "RL":#TRAINS RL to take itself into consieration when planning
                        self.current_player.check_ready_to_accuse()
                        self.metrics.outcome = 'record turn'
                        self.metrics.process_current_data(turns=self.turns,board=self.clue_game.board_nav,player_cards=self.current_player.all_players_possible_cards,confidential=self.current_player.case_file_who_prob+self.current_player.case_file_where_prob+self.current_player.case_file_weapon_prob,player_locations=self.clue_game.current_pos,character=self.current_player.name,stored_player_guesses=self.current_player.stored_player_guesses,final_player_certainty=self.current_player.certainty,reward=self.current_player.earned_rewards)#Gather training data for heuristic evaluation function
                #####################################################################
                else:#logic bot
                    if TRAIN and self.current_player.name == "Mrs. Peacock":
                        self.current_player.check_ready_to_accuse()
                        self.metrics.outcome = 'record turn'
                        self.metrics.process_current_data(turns=self.turns,board=self.clue_game.board_nav,player_cards=self.current_player.all_players_possible_cards,confidential=self.current_player.case_file_who_prob+self.current_player.case_file_where_prob+self.current_player.case_file_weapon_prob,player_locations=self.clue_game.current_pos,character=self.current_player.name,stored_player_guesses=self.current_player.stored_player_guesses,final_player_certainty=self.current_player.certainty,reward=self.current_player.earned_rewards)#Gather training data for heuristic evaluation function
                    self.clue_game.closest_auto_move(self.current_player.name,self.current_player.case_file_where_prob,player_type='logic')
            #####################################################################
            
            self.update_coords()
            self.current_player.room = self.clue_game.player_location_tile(self.current_player.name)
            passed_interogations = []
            if self.current_player.room in self.clue_game.room_list:
                if OLD_LOGIC:
                    where_guess = random.choice(self.current_player.case_file_where_prob)
                else:
                    where_guess = [self.current_player.room]
                who_guess = random.choice(self.current_player.case_file_who_prob)
                weapon_guess = random.choice(self.current_player.case_file_weapon_prob)
                if self.current_player.player_type=="human":
                    where_guess = input([self.current_player.case_file_where_prob.index(i) for i in self.current_player.case_file_where_prob])
                    who_guess = input([self.current_player.case_file_who_prob.index(i) for i in self.current_player.case_file_who_prob])
                    weapon_guess = input([self.current_player.case_file_weapon_prob.index(i) for i in self.current_player.case_file_weapon_prob])
                guess = [who_guess, where_guess, weapon_guess]

                passed_interogations = []
                answeree = []
                for interogated_player in self.player_order:
                    if interogated_player.name != self.current_player.name:
                        interogated_player.hand = self.clue_game.hands[interogated_player.name]
                        guess_answer = interogated_player.guessing(guess) #
                        if len(guess_answer) > 0:
                            answeree = interogated_player.name
                            break
                        passed_interogations.append(interogated_player.name)

                #   loop through player order list and check hands for guessed cards
                self.current_player.update_casefile_probs(guess_maker=self.current_player.name,cards=guess,guess_answers=guess_answer,passed_interogations=passed_interogations,answeree=answeree)

                for player in self.player_order:
                    player.update_casefile_probs(guess_maker=self.current_player.name,cards=guess,guess_answers=guess_answer,passed_interogations=passed_interogations,answeree=answeree)

            #Check if the current player wants to accuse.
            accusation_validity = self.current_player.accuse(confidential=self.clue_game.confidential)


            #print more metrics if set to true
            if DISPLAY_STATS:
                print("Current player is " + self.current_player.name)
                #print('Case File confidential: ', self.clue_game.confidential)

                print('Rolled: ', self.current_player.moves_left)
                print(' --- History: ', self.clue_game.hist[self.current_player.name],' --- Last room: ', self.clue_game.last_room[self.current_player.name],' --- Position value: ', self.clue_game.player_location_tile(self.current_player.name) ,' --- Moves left: ', self.clue_game.moves_left)
                #print("guess",guess ,'--->',guess_answer, '--->',passed_interogations)
                print(f"{round(self.current_player.certainty*100,2)}% chance of correctly guessing. Who {round((1/len(self.current_player.case_file_who_prob))*100,2)}%, Where {round((1/len(self.current_player.case_file_where_prob))*100,2)}%, weapon {round((1/len(self.current_player.case_file_weapon_prob))*100,2)}%" )

            if accusation_validity:#Pauses the game during a special case
                self.game_over = True
                self.metrics.player_win_times[self.current_player.name] += 1
                self.metrics.total_games +=1

                if TRAIN:
                    self.metrics.outcome = self.current_player.name
                    self.metrics.process_current_data(turns=turns,board=self.clue_game.board_nav,player_cards=self.current_player.all_players_possible_cards,confidential=self.current_player.case_file_who_prob+self.current_player.case_file_where_prob+self.current_player.case_file_weapon_prob,player_locations=self.clue_game.current_pos,character=self.current_player.name,stored_player_guesses=self.current_player.stored_player_guesses,final_player_certainty=self.current_player.certainty,reward=self.current_player.earned_rewards)#Label training data for heuristic evaluation function
        if self.game_over == False:
            self.player_order.append(self.current_player)  # move current player to end of player_order list
            self.current_player = self.player_order.popleft()

    e = t()      
class DEEPClueApp(App):
    
    def build(self):
        game = MainWidget()
        return game
       
if __name__ == '__main__':
    if GUI:
        DEEPClueApp().run()
    else: 
        who = ['Miss Scarlet', 'Colonel Mustard', 'Mrs. White',  'Mr. Green',  'Mrs. Peacock', 'Professor Plum']
        where = ['Study', 'Hall', 'Lounge', 'Dining Room', 'Kitchen', 'Ballroom', 'Conservatory', 'Billiard Room', 'Library']
        weapon = ['Rope', 'Lead pipe', 'Knife', 'Wrench', 'Candlestick', 'Pistol']
        all_cards = who + where + weapon
        metrics = metrics_and_tables(all_cards)
        metrics.outcome = 'record turn'
        room_dict = {'dh':'Hall', 'ds':'Study', 'dli':'Library', 'dbi':'Billiard Room', 'dba':'Ballroom', 'dc':'Conservatory','dk':'Kitchen', 'dd':'Dining Room', 'dlo':'Lounge'}

        for _ in range(SETS_OF_ROUNDS):
            for game in range(ROUNDS):
                turns = 0
                player_num = 6
                # initialize/reset board, players, hands, probabilities
                game_over = False
                clue_game = setup_board()
                clue_game.card_setup()
                if HUMAN_PLAYER:
                    player1 = Player(name="Miss Scarlet", current_pos=[1, 17], hand=clue_game.hands["Miss Scarlet"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Miss Scarlet"],all_cards=all_cards,player_type='human')
                else: 
                    player1 = Player(name="Miss Scarlet", current_pos=[1, 17], hand=clue_game.hands["Miss Scarlet"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Miss Scarlet"],all_cards=all_cards,player_type='logic')
                player2 = Player(name="Colonel Mustard", current_pos=[8, 24], hand=clue_game.hands["Colonel Mustard"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Colonel Mustard"],all_cards=all_cards)
                player3 = Player(name="Mrs. White", current_pos=[25, 15], hand=clue_game.hands["Mrs. White"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mrs. White"],all_cards=all_cards)
                player4 = Player(name="Mr. Green", current_pos=[25, 10], hand=clue_game.hands["Mr. Green"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mr. Green"],all_cards=all_cards)
                if RL_AGENT_AS_MRS_PEACOCK:
                    player5 = Player(name="Mrs. Peacock", current_pos=[19, 1], hand=clue_game.hands["Mrs. Peacock"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mrs. Peacock"],all_cards=all_cards,player_type='RL')
                else: 
                    player5 = Player(name="Mrs. Peacock", current_pos=[19, 1], hand=clue_game.hands["Mrs. Peacock"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Mrs. Peacock"],all_cards=all_cards,player_type='logic')
                player6 = Player(name="Professor Plum", current_pos=[6, 1], hand=clue_game.hands["Professor Plum"], who=who, where=where, weapon=weapon,accuse_threshold=metrics.player_thresholds["Professor Plum"],all_cards=all_cards)

                # using a queue data struct - we can "pop" the first index and then append it to cycle through the list of players and maintain order
                player_order = deque([player1, player2, player3, player4, player5, player6])
                current_player = player_order.popleft()
            
                # use while loop to "play" the game:
                s = t()
                while not game_over:
                    turns = turns + 1
                    clue_game.reset_hist(current_player.name)
                    if not current_player.loser:
                        current_player.earned_rewards += int(current_player.earned_rewards*.94)
                        #move
                        clue_game.moves_left = current_player.dice_roll()
                        moves = 0
                        role = clue_game.moves_left
                        if current_player.player_type == 'human':#human player
                            while clue_game.moves_left > 0 and moves < 40:
                                #############################################################
                                
                                possible_moves = []#gets 4 possible legal moves
                                x,y = clue_game.current_pos[current_player.name]
                                for i in [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]:
                                    if clue_game.board_nav[i[0]][i[1]] not in ['x'] + [clue_game.last_room[current_player.name]] and i not in clue_game.blocking_locations():
                                        possible_moves.append(i)
                                human_move = None
                                i = int(input(f"Input one of these: {possible_moves} = {[i for i in range(len(possible_moves))]} or 5 to enter room: "))
                                if i != 5:
                                    human_move = possible_moves[i]
                                if  [x-1,y]   == human_move: human_move = 1
                                elif [x+1,y]  == human_move: human_move = 2
                                elif [x,y-1]  == human_move: human_move = 3
                                elif [x,y+1]  == human_move: human_move = 4
                                print(human_move)
                                clue_game.navigate(human_move,current_player.name)
                                if human_move == None:
                                    print("Out of possible moves")
                                moves += 1
                                clue_game.moves_left = role - len(clue_game.hist[current_player.name])
                            ##############################################################
                        else:#none human player
                            if current_player.player_type == 'RL':
                                current_player.check_ready_to_accuse()
                                possible_moves = []#gets 4 possible legal moves
                                x,y = clue_game.current_pos[current_player.name]
                                l = copy.deepcopy(clue_game.current_pos)
                                player_blocking_locations = clue_game.blocking_locations()
                                room_dict = {'dh':'Hall', 'ds':'Study', 'dli':'Library', 'dbi':'Billiard Room', 'dba':'Ballroom', 'dc':'Conservatory','dk':'Kitchen', 'dd':'Dining Room', 'dlo':'Lounge','none':''}
                                dict_of_doors_to_be_pruned ={'Study' : [[5, 7]], 'Hall' : [[5, 9], [8, 12], [8, 13]], 'Lounge' : [[7, 18]], 'Dining Room' : [[9, 18], [13, 16]], 'Kitchen' : [[18, 20]], 'Ballroom' : [[20, 8], [20, 17], [17, 10], [17, 15]], 'Conservatory' : [[20, 6]], 'Billiard Room' : [[16, 7], [12, 2]], 'Library' : [[9, 8], [12, 4]]}
                                doors = []
                                
                                for set_of_rooms in current_player.case_file_where_prob:
                                    if set_of_rooms != room_dict[clue_game.last_room[current_player.name]]:
                                        for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                                            if room not in player_blocking_locations:
                                                doors.append(room)
                                if len(doors) == 0:
                                    for set_of_rooms in dict_of_doors_to_be_pruned:
                                        if set_of_rooms != room_dict[clue_game.last_room[current_player.name]]:
                                            for room in dict_of_doors_to_be_pruned[set_of_rooms]:
                                                if room not in player_blocking_locations:
                                                    doors.append(room)
                                
                                metrics.outcome = 'RL'
                                for coordinates in doors:
                                        possible_moves.append(coordinates)
                                        l[current_player.name] = coordinates#alter the reported current location of the RL player for all options then pass into a dataset
                                        metrics.process_current_data(turns=turns,board=clue_game.board_nav,player_cards=current_player.all_players_possible_cards,confidential=current_player.case_file_who_prob+current_player.case_file_where_prob+current_player.case_file_weapon_prob,player_locations=l,character=current_player.name,stored_player_guesses=current_player.stored_player_guesses,final_player_certainty=current_player.certainty,reward=current_player.earned_rewards)
                                metrics.outcome = 'RL dataset'#create dataset
                                metrics.process_current_data(turns=turns,board=clue_game.board_nav,player_cards=current_player.all_players_possible_cards,confidential=current_player.case_file_who_prob+current_player.case_file_where_prob+current_player.case_file_weapon_prob,player_locations=l,character=current_player.name,stored_player_guesses=current_player.stored_player_guesses,final_player_certainty=current_player.certainty,reward=current_player.earned_rewards)
                                #check for rooms
                                clue_game.closest_auto_move(current_player.name,current_player.case_file_where_prob,player_type='RL',rl_input=metrics.predict_dataset,possible_moves=possible_moves)
                                if TRAIN and current_player.name == "Mrs. Peacock":#TRAINS RL to take itself into consieration when planning
                                    current_player.check_ready_to_accuse()
                                    metrics.outcome = 'record turn'
                                    metrics.process_current_data(turns=turns,board=clue_game.board_nav,player_cards=current_player.all_players_possible_cards,confidential=current_player.case_file_who_prob+current_player.case_file_where_prob+current_player.case_file_weapon_prob,player_locations=clue_game.current_pos,character=current_player.name,stored_player_guesses=current_player.stored_player_guesses,final_player_certainty=current_player.certainty,reward=current_player.earned_rewards)#Gather training data for heuristic evaluation function
                            #####################################################################
                            else:#logic bot
                                if TRAIN and current_player.name == "Mrs. Peacock":
                                    current_player.check_ready_to_accuse()
                                    metrics.outcome = 'record turn'
                                    metrics.process_current_data(turns=turns,board=clue_game.board_nav,player_cards=current_player.all_players_possible_cards,confidential=current_player.case_file_who_prob+current_player.case_file_where_prob+current_player.case_file_weapon_prob,player_locations=clue_game.current_pos,character=current_player.name,stored_player_guesses=current_player.stored_player_guesses,final_player_certainty=current_player.certainty,reward=current_player.earned_rewards)#Gather training data for heuristic evaluation function
                                clue_game.closest_auto_move(current_player.name,current_player.case_file_where_prob,player_type='logic')
                        #####################################################################
                        
                        current_player.room = clue_game.player_location_tile(current_player.name)
                        passed_interogations = []

                        if current_player.room in clue_game.room_list:
                            if OLD_LOGIC:
                                where_guess = random.choice(current_player.case_file_where_prob)
                            else:
                                 where_guess = [current_player.room]
                            who_guess = random.choice(current_player.case_file_who_prob)
                            weapon_guess = random.choice(current_player.case_file_weapon_prob)
                            if current_player.player_type=="human":
                                where_guess = where_guess = [current_player.room]#room_dict[current_player.room]
                                who_guess = input([current_player.case_file_who_prob.index(i) for i in current_player.case_file_who_prob])
                                weapon_guess = input([current_player.case_file_weapon_prob.index(i) for i in current_player.case_file_weapon_prob])
                            guess = [who_guess, where_guess, weapon_guess]

                            passed_interogations = []
                            answeree = []
                            for interogated_player in player_order:
                                if interogated_player.name != current_player.name:
                                    interogated_player.hand = clue_game.hands[interogated_player.name]
                                    guess_answer = interogated_player.guessing(guess) #
                                    if len(guess_answer) > 0:
                                        answeree = interogated_player.name
                                        break
                                    passed_interogations.append(interogated_player.name)

                            #   loop through player order list and check hands for guessed cards
                            current_player.update_casefile_probs(guess_maker=current_player.name,cards=guess,guess_answers=guess_answer,passed_interogations=passed_interogations,answeree=answeree)
                                
                            for player in player_order:
                                player.update_casefile_probs(guess_maker=current_player.name,cards=guess,guess_answers=guess_answer,passed_interogations=passed_interogations,answeree=answeree)

                        #Check if the current player wants to accuse.
                        accusation_validity = current_player.accuse(confidential=clue_game.confidential)

                        #print more metrics if set to true
                        if DISPLAY_STATS:
                            print("Current player is " + current_player.name)
                            #print('Case File confidential: ', clue_game.confidential)

                            print('Rolled: ', current_player.moves_left)
                            print(' --- History: ', clue_game.hist[current_player.name],' --- Last room: ', clue_game.last_room[current_player.name],' --- Position value: ', clue_game.player_location_tile(current_player.name) ,' --- Moves left: ', clue_game.moves_left,"Goal:",clue_game.goal[current_player.name])
                            #print("guess",guess ,'--->',guess_answer, '--->',passed_interogations)
                            print(f"{round(current_player.certainty*100,2)}% chance of correctly guessing. Who {round((1/len(current_player.case_file_who_prob))*100,2)}%, Where {round((1/len(current_player.case_file_where_prob))*100,2)}%, weapon {round((1/len(current_player.case_file_weapon_prob))*100,2)}%" )

                        if accusation_validity:#Pauses the game during a special case 
                            game_over = True
                            metrics.player_win_times[current_player.name] += 1
                            metrics.total_games +=1 
                            
                            if TRAIN:
                                metrics.outcome = current_player.name
                                metrics.process_current_data(turns=turns,board=clue_game.board_nav,player_cards=current_player.all_players_possible_cards,confidential=current_player.case_file_who_prob+current_player.case_file_where_prob+current_player.case_file_weapon_prob,player_locations=clue_game.current_pos,character=current_player.name,stored_player_guesses=current_player.stored_player_guesses,final_player_certainty=current_player.certainty,reward=current_player.earned_rewards)#Gather training 
                            break

                    player_order.append(current_player)  # move current player to end of player_order list
                    current_player = player_order.popleft() 

                e = t()
                if DISPLAY_STATS == False:
                        print("game",game,[f"{i}: {metrics.player_win_times[i]}: {round(metrics.player_win_times[i]/(game+1)*100)}%" for i in [name for name in metrics.player_win_times]],"game time:",round(abs(e - s),3),end='\r     ')
                    
            #if DISPLAY_STATS: 
            print("game",game,[f"{i}: {metrics.player_win_times[i]}: {round(metrics.player_win_times[i]/(ROUNDS)*100)}%" for i in [name for name in metrics.player_win_times]],"game time:",round(abs(e - s),3),end='\r     ')  
            metrics.reset()

            if TRAIN:
                metrics.outcome = 'Conv to dataset'
                metrics.process_current_data(turns=turns,board=clue_game.board_nav,player_cards=current_player.all_players_possible_cards,confidential=current_player.case_file_who_prob+current_player.case_file_where_prob+current_player.case_file_weapon_prob,player_locations=clue_game.current_pos,character=current_player.name,stored_player_guesses=current_player.stored_player_guesses,final_player_certainty=current_player.certainty)
                if metrics.dataset != None: #First run, only collect data for usage as a test ds
                    metrics.model(loss='mse',activation='relu')                  
    if TRAIN and GUI == False:
        metrics.monte_carlo_heuristic.save(f'{current_path}/clue1.h5', overwrite=True, include_optimizer=True, save_traces=True)