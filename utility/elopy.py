import pickle
import os.path

class Elo:
    def __init__(self, base_rating=1500):
        self.base_rating = base_rating
        self.players = []

    def __str__(self):
        tb = self.getRatingList
        tb = sorted(tb, key=lambda x: x[1])[::-1]

        string = '-'*(20+3) + '\n'
        format_string = '|{:>15}|{:>5}|\n'
        for name, rate in tb:
            rate = int(rate)
            string += format_string.format(name, rate)
        string += '-'*(20+3)
        return string

    def save(self, filename='elo.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.players, file)

    def load(self, filename='elo.pkl'):
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                self.players = pickle.load(file)
            print(self.players)
            print(self.__str__())

    def getPlayer(self, name):
        for player in self.players:
            if player.name == name:
                return player
        return None

    def contains(self, name):
        for player in self.players:
            if player.name == name:
                return True
        return False

    def addPlayer(self, name, rating=None):
        if rating == None:
            rating = self.base_rating

        self.players.append(_Player(name=name,rating=rating))

    def removePlayer(self, name):
        self.getPlayerList.remove(self.getPlayer(name))


    def recordMatch(self, name1, name2, winner=None, draw=False, verbose=False):
        player1 = self.getPlayer(name1)
        player2 = self.getPlayer(name2)

        expected1 = player1.compareRating(player2)
        expected2 = player2.compareRating(player1)
        
        k = 16

        rating1 = player1.rating
        rating2 = player2.rating

        if draw:
            score1 = 0.5
            score2 = 0.5
        elif winner == name1:
            score1 = 1.0
            score2 = 0.0
        elif winner == name2:
            score1 = 0.0
            score2 = 1.0
        else:
            raise InputError('One of the names must be the winner or draw must be True')

        newRating1 = rating1 + k * (score1 - expected1)
        newRating2 = rating2 + k * (score2 - expected2)

        if newRating1 < 0:
            newRating1 = 0
            newRating2 = rating2 - rating1

        if newRating2 < 0:
            newRating2 = 0
            newRating1 = rating1 - rating2

        if verbose:
            print(f'{player1.name} : {player1.rating} -> {newRating1}')
            print(f'{player2.name} : {player2.rating} -> {newRating2}')
        player1.rating = newRating1
        player2.rating = newRating2

    def getPlayerRating(self, name):
        player = self.getPlayer(name)
        return player.rating

    @property
    def getPlayerList(self):
        return self.players

    @property
    def getRatingList(self):
        lst = []
        for player in self.getPlayerList:
            lst.append((player.name,player.rating))
        return lst

class _Player:
    def __init__(self, name, rating):
        self.name = name
        self.rating = rating

    def compareRating(self, opponent, b=10, alpha=400.0):
        return 1/(1+b**((opponent.rating-self.rating)/alpha))
