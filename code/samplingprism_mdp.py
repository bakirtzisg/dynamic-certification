from __future__ import division
import stormpy
import stormpy.core
import stormpy.logic
import stormpy.pars
import re
import stormpy.examples
import stormpy.examples.files
import time
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from stormpy.utility import ShortestPathsGenerator
import os

# from gurobipy import *


# Approximation of binomial cdf with continuity correction for large n
# n: trials, p: success prob, m: starting successes
def BCDF(p, n, m):
    return 1-CDF((m-0.5-(n*p))/math.sqrt(n*p*(1-p)))
def CDF(x):
    return (1.0 + math.erf(x/math.sqrt(2.0)))/2.0


#loading model and specs
def loader():
    path = "../models/UAV_grid.prism"
    prism_program = stormpy.parse_prism_program(path)
    print("Building model from {}".format(path))
    formula_str = 'Pmax=? [!"Crash" U "Goal"]'
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    options = stormpy.BuilderOptions([properties[0].raw_formula])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    options.set_build_all_reward_models()
    model = stormpy.build_sparse_parametric_model_with_options(prism_program,options)

    print("Model supports parameters: {}".format(model.supports_parameters))
    parameters = model.collect_probability_parameters()
    numstate=model.nr_states
    print("Number of states before bisim:",numstate)
    #    #assert len(parameters) == 2
    print ("Number of params before bisim:",len(parameters))
    #   print(model.model_type)
    # instantiator = stormpy.pars.PDtmcInstantiator(model)
    #print (model.initial_states)
    #gathering parameters in the bisimulated mpdel
    # model = stormpy.perform_bisimulation(model, properties, stormpy.BisimulationType.STRONG)
    # parameters= model.collect_probability_parameters()
    # parameters_rew = model.collect_reward_parameters()
    # parameters.update(parameters_rew)


    numstate=model.nr_states
    print("Number of states after bisim:",numstate)
    # #    #assert len(parameters) == 2
    # print ("Number of params after bisim:",len(parameters))
    return parameters,model,properties

def run_sample(numiter,numsample,thres,direction,parameters,model,properties):
    '''

    :param numiter: number of trials to compute the number of samples that satisfies the probability
    :param numsample: number of sampled pMDPs/pMCs to check
    :param thres: specification threshold
    :param direction: if True, then the spec is \geq, if False, then it is \leq
    :return:
    '''





    #storing the approximate satisfaction probability for each iteration
    # move_ids = [m_s.id for m_s in model.states for s_i in m_s.labels if 'go' in s_i or 'stop' in s_i]
    # action_points = set(range(model.nr_states)) - set(move_ids)
    # actions1 = ['go1','stop1']
    # actions2 = ['go2','stop2']
    #
    # second_car_points = []
    # for m_s in model.states:
    #     flag1 = False
    #     flag2 = True
    #     for s_i in m_s.labels:
    #         if 'go1' in s_i or 'stop1' in s_i:
    #             flag1 = True
    #         if 'go2' in s_i or 'stop2' in s_i:
    #             flag2 = False
    #     if flag1 and flag2:
    #         second_car_points.append(m_s.id)
    # counterarray= [0 for _ in range(numiter)]

    instantiator = stormpy.pars.PMdpInstantiator(model)

    start3 = time.time()
    #for each iteration
    data_out = dict()
    for iter in range(numiter):
        counter=0
        # for each sample
        for i in tqdm(range(int(numsample))):
            point=dict()
            Agent1_pol = dict()
            for x in parameters:
                s = np.random.uniform(0.0, 0.2)
                point[x] = stormpy.RationalRF(s)
                # point2[x] = stormpy.RationalRF(s2)
            #check result
            rational_parameter_assignments = dict([[x, stormpy.RationalRF(val)] for x, val in point.items()])
            instantiated_model = instantiator.instantiate(rational_parameter_assignments)
            time_t1 = time.time()
            result = stormpy.model_checking(instantiated_model, properties[0],extract_scheduler=True)
            print('Synth RT {}'.format(time.time()-time_t1))
            dtmc = instantiated_model.apply_scheduler(result.scheduler)
            data_out.update({s:result.at(0)})
        file_loc = os.getcwd() + '/grid.drn'
        stormpy.export_to_drn(dtmc, file_loc)
        convert_drn_to_dtmc(file_loc)
        dtmc = stormpy.build_model_from_drn(file_loc)
        counter_examples = []
        spg = ShortestPathsGenerator(dtmc, "Crash")
        for k in range(1, 6):
            states = spg.get_path_as_list(k)
            path_valuations = []
            for k_i in reversed(states):
                k_val = dtmc.states[k_i].labels
                path_valuations.append(convert_labels_to_tuple(k_val))
            counter_examples.append(path_valuations)
    return counter_examples,data_out

## Converting the sparse MDP to sparse DTMC for the counterexample paths
def convert_drn_to_dtmc(file_loc):
    a_file = open(file_loc,'r')
    list_of_lines = a_file.readlines()
    list_of_lines[2] = '@type: DTMC\n'
    a_file = open(file_loc,'w')
    a_file.writelines(list_of_lines)
    a_file.close()

def convert_labels_to_tuple(labels):
    hold_list = [0,0,0,0,0]
    for l_i in labels:
        if re.search('rx',l_i):
            hold_list[2] = int(re.search(r'\d+', l_i).group())
        elif re.search('ry',l_i):
            hold_list[3] = int(re.search(r'\d+', l_i).group())
        elif re.search('x_',l_i):
            hold_list[0] = int(re.search(r'\d+', l_i).group())
        elif re.search('y_',l_i):
            hold_list[1] = int(re.search(r'\d+', l_i).group())
        elif re.search('downed',l_i):
            hold_list[4] = 1
    return tuple(hold_list)
            # rational_parameters_oc = dict([[parameter_oc, stormpy.RationalRF(s)]])
            # ins_oc = instant_model.instantiate(rational_parameters_oc)
            # result_oc = stormpy.model_checking(ins_oc, prop[0], extract_scheduler=True)


            # for s_i in action_points:
            #     for l_i in model.states[s_i].labels:
            #         if 's' in l_i and not 'Crash' in l_i:
            #             s_state = l_i
            #         elif 'x' in l_i:
            #             x_state = l_i
            #         elif 'p' in l_i:
            #             p_state = l_i
            #     hold_state = model.states[int(re.findall('\\d+', str(model.states[s_i].actions[result.scheduler.get_choice(s_i).get_deterministic_choice()].transitions))[0])]
            #     next_action = result.scheduler.get_choice(hold_state).get_deterministic_choice()
            #     next_state = model.states[int(re.findall('\\d+', str(hold_state.actions[int(next_action)].transitions))[0])]
            #     if 'Crash' not in next_state.labels and 'Goal' not in next_state.labels:
            #         act_tup = tuple()
            #         act_tup += ([l_ind for l_ind,l_a in enumerate(actions1) if l_a in next_state.labels][0],)
            #         act_tup += ([l_ind for l_ind,l_a in enumerate(actions2) if l_a in next_state.labels][0],)
            #         Agent1_pol.update({(s_state, x_state, p_state): act_tup[0]})
            #     hold_state2 = model.states[int(re.findall('\\d+', str(model.states[s_i].actions[result2.scheduler.get_choice(s_i).get_deterministic_choice()].transitions))[0])]
            #     next_action2 = result2.scheduler.get_choice(hold_state2).get_deterministic_choice()
            #     next_state2 = model.states[int(re.findall('\\d+', str(hold_state2.actions[int(next_action2)].transitions))[0])]
            #     if 'Crash' not in next_state2.labels and 'Goal' not in next_state2.labels:
            #         act_tup2 = tuple()
            #         act_tup2 += ([l_ind for l_ind,l_a in enumerate(actions1) if l_a in next_state2.labels][0],)
            #         act_tup2 += ([l_ind for l_ind,l_a in enumerate(actions2) if l_a in next_state2.labels][0],)
            #         Agent2_pol.update({(s_state,x_state,p_state):act_tup2[1]})

            # for s_i in range(model_one_car.nr_states):
            #     for l_i in model_one_car.states[s_i].labels:
            #         if 's' in l_i and not 'Crash' in l_i:
            #             s_state = l_i
            #         elif 'x' in l_i:
            #             x_state = l_i
            #         elif 'p' in l_i:
            #             p_state = l_i
            #     for x_state in range(6):
            #         Agent1_pol.update({(s_state,'x{}'.format(x_state),p_state):result_oc.scheduler.get_choice(s_i).get_deterministic_choice()%2})

            # for s_i in second_car_points:
            #     for l_i in model.states[s_i].labels:
            #         if 's' in l_i and not 'Crash' in l_i and not 'stop' in l_i:
            #             s_state = l_i
            #         elif 'x' in l_i:
            #             x_state = l_i
            #         elif 'p' in l_i and not 'stop' in l_i:
            #             p_state = l_i
            #     Agent2_pol.update({(s_state,x_state,p_state):result2.scheduler.get_choice(s_i).get_deterministic_choice()})

            # result = stormpy.model_checking(instantiated_model, properties[0]).at(instantiated_model.initial_states[0])
            #append the counter according to the spec

def compute_avg_satprob(counterarray,N,eps,flag):
    '''
    :param counterarray: approximate satisfaction probs
    :param N: number of samples
    :param eps:
    :param direction: if True, then the spec is \geq, if False, then it is \leq

    :return:
    '''
    #storing probabilities for each iteration
    valuearray = np.zeros(len(counterarray))

    for iters in range(len(counterarray)):
        print(counterarray[iters])

        val = 0
        val2 = 0
        #compute constraints to remove in the LP
        if flag:
            removeconstraints = int(N * (counterarray[iters]))
        else:
            removeconstraints = int(N * (1 - counterarray[iters]))
        # print(N,removeconstraints,eps)

        val3 = BCDF(eps, N, removeconstraints)

        valuearray[iters] = val3

    print("probability of violating the spec less then the threshold for each iter:")
    print(valuearray)
    print("average value of the array:")
    print(np.mean(valuearray))



parameters,model,properties=loader()
numiter=1
numsample=100
threshold=0.75
direction=True
counter_array,data_out=run_sample(numiter,numsample,threshold,direction,parameters,model,properties)
fig = plt.figure()
ax = plt.axes()
xline = []
yline = []
# zline = []
for x in data_out:
    xline.append(x)
    yline.append(data_out[(x)])
    # yline.append(y)

a_li = np.asarray([xline,yline])
np.savetxt('Grid2.csv',a_li.T,delimiter=',')
ax.scatter(xline,yline)
ax.set_xlabel(r'$p_1$')
ax.set_ylabel(r'$Pr_{max}[\neg Crash~{\sf U}~Goal]$')
plt.savefig('Grid2.png')
