from os.path import join
import helpers
import textwrap
import load_data2 as ld2
import match_chips2 as mc2
import numpy as np
import os
import params
import report_results2 as rr2
import sys
import draw_func2 as df2
import itertools

def reload_module():
    import imp
    import sys
    imp.reload(sys.modules[__name__])

def gen_subset_split(full_set, N, K):
    np.random.seed(0) # repeatibility
    seen = set([])
    split_list = []
    for kx in xrange(K):
        np.random.shuffle(full_set)
        failsafe = 0
        while True: 
            np.random.shuffle(full_set)
            subset = tuple(full_set[0:N])
            if not subset in seen: 
                seen.add(subset)
                compliment = tuple(np.setdiff1d(full_set, subset))
                yield (compliment, subset)
                break
            failsafe += 1
            if failsafe > 100:
                break

def run_experiment2(hs, test_sample_cx, train_sample_cx):
    if not hs in vars() or hs is None:
        hs = ld2.HotSpotter()
        hs.load_tables(ld2.DEFAULT)
    # Try and just use the cache if possible
    hs.set_test_train(test_sample_cx, train_sample_cx)
    qcx2_res, dirty_samp = mc2.load_cached_matches(hs)
    if len(dirty_samp) > 0:
        if hs.matcher is None:
            # Well, if we can't use the cache
            print('[expt] LAZY LOADING FEATURES AND MATCHER')
            print('[expt] There are %d dirty queries' % len(dirty_samp))
            hs.load_chips()
            hs.load_features()
            hs.load_matcher()
        qcx2_res = mc2.run_matching(hs, qcx2_res, dirty_samp)
    allres = rr2.report_all(hs, qcx2_res)
    
def leave_N_names_out(N):
    hs = ld2.HotSpotter()
    hs.load_tables(ld2.DEFAULT)
    hs.load_chips()
    hs.load_features()
    nx2_name = hs.tables.nx2_name
    cx2_nx  = hs.tables.cx2_nx
    nx2_cxs = np.array(hs.get_nx2_cxs())
    nx2_nChips = np.array(map(len, nx2_cxs))
    nx2_nChips[0:3] = 0 # remove uniden names
    all_nxs,  = np.where((nx2_nChips > 0))
    M = len(all_nxs)
    N = M//2 
    K = 3
    subset_gen = gen_subset_split(all_nxs, N, K)
    mylist = []
    for kx in xrange(K):
        print('***************')
        print('[expt] Leave N out iteration: %d/%d' % (kx, K))
        print('***************')
        (nxs1, nxs2) = subset_gen.next()
        test_sample_cx = np.hstack(nx2_cxs[np.array(nxs1)])
        train_sample_cx = np.hstack(nx2_cxs[np.array(nxs2)])
        hs.set_samples(test_sample_cx, train_sample_cx)
        hs.load_matcher()
        qcx2_res = mc2.run_matching(hs)
        mylist.append(qcx2_res)
    
    #do with TF-IDF on the zebra data set. 
    #Let M be the total number of *animals* (not images and not chips) in an experimental data set. 
    #Do a series of leave-N-out (N >= 1) experiments on the TF-IDF scoring,
    #where the "left out" N are N different zebras, 
    #so that there are no images of these zebras in the images used to form the vocabulary.
    #The vocabulary is formed from the remaining M-N animals.
    #Test how well TF-IDF recognition does with these N animals. 
    #Repeat for different subsets of N animals.

def oxford_philbin07(hs=None):
    params.__MATCH_TYPE__        = 'bagofwords'
    params.__BOW_NUM_WORDS__     = [1e4, 2e4, 5e4, 1e6, 1.25e6][3]
    params.__NUM_RERANK__        = [100, 200, 400, 800, 1000][4]
    params.__CHIP_SQRT_AREA__    = None

    params.__XY_THRESH__         = 0.01
    params.__SCALE_THRESH_HIGH__ = 8
    params.__SCALE_THRESH_LOW__  = 0.5
    params.BOW_AKMEANS_FLANN_PARAMS = dict(algorithm='kdtree',
                                               trees=8, checks=64)
    # I'm not sure if checks parameter is set correctly
    db_dir = ld2.OXFORD
    if hs is None:
        hs = ld2.HotSpotter()
        hs.load_tables(db_dir)
        # Use the 55 cannonical test cases 
        hs.split_test_train_at(55)
    # Try and just use the cache if possible
    # TODO: Make all tests run like this by default (lazy loading)
    qcx2_res, dirty_samp = mc2.load_cached_matches(hs)
    if len(dirty_samp) > 0:
        if hs.matcher is None:
            # Well, if we can't use the cache
            print('[expt] LAZY LOADING FEATURES AND MATCHER')
            print('[expt] There are %d dirty queries' % len(dirty_samp))
            hs.load_chips()
            hs.load_features()
            hs.split_test_train_at(55)
            hs.load_matcher()
            #print('[expt] Database shape: '+str(np.vstack(hs.feats.cx2_desc[db_sample_cx]).shape))
        qcx2_res = mc2.run_matching(hs, qcx2_res, dirty_samp)
    allres = rr2.report_all(hs, qcx2_res, oxford=True)
    return locals()

def oxford_bow():
    params.__MATCH_TYPE__     = 'bagofwords'
    params.__CHIP_SQRT_AREA__ = None
    params.__BOW_NUM_WORDS__  = [1e4, 2e4, 5e4, 1e6, 1.25e6][3]
    db_dir = ld2.OXFORD
    hs = ld2.HotSpotter()
    hs.load_tables(db_dir)
    hs.load_chips()
    hs.load_features(load_desc=False)
    hs.set_sample_subset(55, None)
    assert min(hs.test_sample_cx) == 55 and max(hs.test_sample_cx) == 5117
    qcx2_res = mc2.run_matching(hs)
    hs.free_some_memory()
    allres = rr2.report_all(hs, qcx2_res, oxford=False, stem=False, matrix=False)
    return locals()

def oxford_vsmany():
    params.__MATCH_TYPE__     = 'vsmany'
    params.__CHIP_SQRT_AREA__ = None
    db_dir = ld2.OXFORD
    hs = ld2.HotSpotter(db_dir, samples_subset=(55,None))
    qcx2_res = mc2.run_matching(hs)
    allres = rr2.report_all(hs, qcx2_res, oxford=False, stem=False, matrix=False)
    return locals()
    
def mothers_vsmany():
    params.__MATCH_TYPE__     = 'vsmany'
    return run_experiment()

def mothers_bow():
    params.__MATCH_TYPE__     = 'bagofwords'
    return run_experiment()

def tweak_params(expt_func=None):
    if not 'expt_func' in vars() or expt_func is None:
        expt_func = run_experiment
    xy_thresh_tweaks  = [.05, .01, .005, .001]
    scale_low_tweaks  = [.75, .5, .25]
    scale_high_tweaks = [1.5, 2, 8]
    gen_ = itertools.product(xy_thresh_tweaks, scale_high_tweaks, scale_low_tweaks)
    parameter_list = [tup for tup in gen_]

    total_tests = len(parameter_list)
    result_map = {}
    hs = None
    for count, tup in enumerate(parameter_list):
        print('**********************************************************')
        print('**********************************************************')
        print('========================')
        print('[expt] tweak_params(%d/%d)> param tweak %r ' % (count, total_tests, tup,))
        print('========================')
        rss = helpers.RedirectStdout()
        rss.start()
        xy_thresh, scale_thresh_high, scale_thresh_low = tup
        params.__XY_THRESH__         = xy_thresh
        params.__SCALE_THRESH_LOW__  = scale_thresh_low
        params.__SCALE_THRESH_HIGH__ = scale_thresh_high
        expt_locals = expt_func(hs)
        if 'hs' in expt_locals:
            hs = expt_locals['hs']
        result_map[tup] = expt_locals['allres']
        rss.stop()
    return locals()

def tweak_params_philbin():
    return tweak_params(oxford_philbin07)

def demo():
    pass
#ld2.DEFAULT

def run_experiment(hs=None):
    'Runs experiment and dumps results. Returns locals={qcx2_res, hs}'
    if hs is None:
        db_dir = ld2.DEFAULT
        hs = ld2.HotSpotter(db_dir)
    print('======================')
    print('[expt] Running Experiment on hs:\n'+str(hs))
    print('[expt] Params: \n'+ helpers.indent(params.param_string()))
    print('======================')
    qcx2_res = mc2.run_matching(hs)
    allres   = rr2.report_all(hs, qcx2_res)
    return locals()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('\n\n\n[expt] __main__ = experiments.py')
    print('[expt] sys.argv = %r' % sys.argv)

    # Default to run_experiment
    expt_func = run_experiment

    arg_map = {
        'philbin'        : oxford_philbin07,
        'oxford-bow'     : oxford_bow,
        'oxford-vsmany'  : oxford_vsmany,
        'mothers-bow'    : mothers_bow,
        'mothers-vsmany' : mothers_vsmany,
        'default'        : run_experiment, 
        'tweak'          : tweak_params,
        'tweak-philbin'  : tweak_params_philbin}

    print ('[expt] Valid experiments are:\n    '+ '\n    '.join(arg_map.keys()))
    argv = sys.argv

    # Change based on user input
    has_arg = False
    for arg in argv:
        if arg in arg_map.keys():
            print('[expt] Running '+str(arg))
            expt_func = arg_map[arg]

    # Do the experiment
    expt_locals = expt_func()
    hs = expt_locals['hs']
    if 'allres' in expt_locals.keys():
        qcx2_res = expt_locals['qcx2_res']
        allres = expt_locals['allres']
        print(allres)

    exec(df2.present())
