import numpy as np
from sklearn.model_selection import cross_val_score
import random


class Chromosome:
    def __init__(self, hp, features, fitness, param_list):
        self.hpGene = hp
        self.featGene = features
        self.fitness = fitness
        self.param_list = param_list

    def __str__(self):
        return "Hyper_Params: {}\nFeature_Fitness: {:.4f}\tModel_Fitness: {:.4f}+/-{:.4f}".\
            format(dict(zip(self.param_list.keys(),self.hpGene)),self.fitness['fit_feat'],self.fitness['fit_mod'],\
                   self.fitness['fit_stdev'])

    def __lt__(self, other):
        if self.fitness['fit_mod'] < other.fitness['fit_mod']:
            return True
        elif self.fitness['fit_mod'] == other.fitness['fit_mod']:
            if self.fitness['fit_stdev'] > other.fitness['fit_stdev']:
                return True
            elif self.fitness['fit_stdev'] == other.fitness['fit_stdev']:
                if self.fitness['fit_feat'] < other.fitness['fit_feat']:
                    return True
                return False
            return False
        return False

    def __eq__(self, other):
        if self.fitness['fit_mod'] == other.fitness['fit_mod'] and \
                self.fitness['fit_stdev'] == other.fitness['fit_stdev'] and \
                self.fitness['fit_feat'] == other.fitness['fit_feat']:
            return True
        return False


def _sampler(samp):
    if samp[0].lower() == 'int':
        return np.random.randint(samp[1], samp[2], 1)[0]

    elif samp[0].lower() == 'float':
        return np.random.uniform(samp[1], samp[2], 1)[0]

    elif samp[0].lower() == 'list':
        return samp[1:][np.random.randint(0,len(samp[1:]),1)[0]]


def _fitness(X, Y, mod, cv, scoring, param_dict, hpGene, featGene):

    up_param_dict = dict(zip(param_dict.keys(), hpGene))

    cv_score =  cross_val_score(estimator=mod.set_params(**up_param_dict), X=X.loc[:,(featGene==1)], y=Y,
                                       cv = cv, scoring = scoring, n_jobs=1)
    fit_mod = np.mean(cv_score)
    fit_stdev = np.std(cv_score)

    fit_feat = 1.0 - (sum(featGene)/len(featGene))

    fit_dict = {'fit_mod' : fit_mod,
                'fit_feat': fit_feat,
                'fit_stdev': fit_stdev}

    return fit_dict


def crossover(X, Y, mod, cv, scoring, parent_dict):

    random.seed()

    rand_hp = np.random.randint(0,2,len(parent_dict[1].hpGene))
    rand_feat = np.random.randint(0,2,len(parent_dict[2].featGene))

    hp = [(parent_dict[1].hpGene[i]+parent_dict[2].hpGene[i])/2 \
              if type(parent_dict[1].hpGene[i]) == type(0.1) and rand_hp[i] else \
              int(round((parent_dict[1].hpGene[i]+parent_dict[2].hpGene[i])/2))\
                  if type(parent_dict[1].hpGene[i]) == type(1) and rand_hp[i] else \
                  parent_dict[rand_hp[i]+1].hpGene[i] for i in range(len(rand_hp))]

    feat = (rand_feat * parent_dict[1].featGene) + (np.abs(1-rand_feat) * parent_dict[2].featGene)

    geneFitness = _fitness(X, Y, mod, cv, scoring, parent_dict[1].param_list, hp, feat)

    return Chromosome(hp, feat, geneFitness, parent_dict[1].param_list)


def generate_parent(X, Y, mod, param_dict, cv, scoring):

    random.seed()
    featGene = np.random.randint(0,2,X.shape[1])

    '''Generate random hyper-parameters'''
    hpGene = [_sampler(param_dict[i]) for i in param_dict.keys()]

    '''Get fitness for parent'''
    geneFitness = _fitness(X, Y, mod, cv, scoring, param_dict, hpGene, featGene)

    return Chromosome(hpGene,featGene,geneFitness,param_dict)

