# -*- coding: utf-8 -*-

import pickle as pkl

# NVGaze
nv_subs_train = ['NVIDIAAR_{}'.format(2500)]
nv_subs_test = ['NVIDIAAR_{}'.format(11200)]

# OpenEDS
openeds_train = ['OpenEDS_{}'.format(2500)]
openeds_test = ['OpenEDS_{}'.format(11200)]

# LPW
lpw_subs_train = ['LPW_{}'.format(2500)]
lpw_subs_test = ['LPW_{}'.format(11200)]

# # Fuhl
fuhl_subs_train = ['Fuhl_{}'.format(2500)]
fuhl_subs_test = ['Fuhl_{}'.format(11200)]


# S-General
riteyes_subs_train = ['riteyes_general_{}'.format(i+1) for i in range(0, 24)]
riteyes_subs_test = ['riteyes_general_{}'.format(i+1) for i in range(23, 24)]

DS_train = {'NVGaze':nv_subs_train,
            'OpenEDS':openeds_train,
            'LPW':lpw_subs_train,
            'Fuhl':fuhl_subs_train,
            'riteyes_general': riteyes_subs_train}

DS_test = {'NVGaze':nv_subs_test,
            'OpenEDS':openeds_test,
            'LPW':lpw_subs_test,
            'Fuhl':fuhl_subs_test,
            'riteyes_general': riteyes_subs_test}

DS_selections = {'train': DS_train,
                 'test' : DS_test}

pkl.dump(DS_selections, open('dataset_selections.pkl', 'wb'))
