import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import copy
from utils import get_data_loader,checkattr
from data.manipulate import SubDataset, MemorySetDataset
from models.cl.continual_learner import ContinualLearner
from eval import evaluate
from models.utils import loss_functions as lf
from torch.nn import functional as F
import logging
import eval.precision_recall as pr
from visual.visual_plt import plot_pr_curves
import utils

def train(model, train_loader, iters, loss_cbs=list(), eval_cbs=list()):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data from [train_loader].

    [model]             model to optimize
    [train_loader]      <dataloader> for training [model] on
    [iters]             <int> (max) number of iterations (i.e., batches) to train for
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model on separate data-set'''
    # device = model._device()
    device='cuda'

    # Create progress-bar (with manual control)
    bar = tqdm.tqdm(total=iters)

    iteration = epoch = 0
    model = model.to(device)
    while iteration < iters:
        epoch += 1

        # Loop over all batches of an epoch
        for batch_idx, (data, y) in enumerate(train_loader):
            iteration += 1
            # print(data.shape)
            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            # loss_dict = model.train_a_batch(data, y=y)
            model.train()
            model.optimizer.zero_grad()
            y_hat = model(data)
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')
            predL.backward()
            model.optimizer.step()
            # Fire training-callbacks (for visualization of training-progress)
            # for loss_cb in loss_cbs:
            #     if loss_cb is not None:
            #         loss_cb(bar, iteration, loss_dict)

            # Fire evaluation-callbacks (to be executed every [eval_log] iterations, as specified within the functions)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration)

            # Break if max-number of iterations is reached
            if iteration == iters:
                bar.close()
                break
def train_old(model, train_loader, iters, loss_cbs=list(), eval_cbs=list()):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data from [train_loader].

    [model]             model to optimize
    [train_loader]      <dataloader> for training [model] on
    [iters]             <int> (max) number of iterations (i.e., batches) to train for
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model on separate data-set'''
    # device = model._device()
    device = model._device()

    # Create progress-bar (with manual control)
    bar = tqdm.tqdm(total=iters)

    iteration = epoch = 0
    while iteration < iters:
        epoch += 1

        # Loop over all batches of an epoch
        for batch_idx, (data, y) in enumerate(train_loader):
            iteration += 1

            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(data, y=y)

            # Fire training-callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(bar, iteration, loss_dict)

            # Fire evaluation-callbacks (to be executed every [eval_log] iterations, as specified within the functions)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration)

            # Break if max-number of iterations is reached
            if iteration == iters:
                bar.close()
                break

#------------------------------------------------------------------------------------------------------------#

def train_cl(model, train_datasets, test_datasets, config, iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), sample_cbs=list(), context_cbs=list(),
             generator=None, gen_iters=0, gen_loss_cbs=list(), first_iters = 0, cycles=0, seed = 0, **kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts.

    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [baseline]            <str>, 'joint': model trained once on data from all contexts
                                 'cummulative': model trained incrementally, always using data all contexts so far
    [generator]           None or <nn.Module>, if separate generative model is trained (for [gen_iters] per context)
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st context)
    ReplayStoredData = ReplayGeneratedData = ReplayCurrentData = False
    previous_model = None

    # Register starting parameter values (needed for SI)
    if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
        model.register_starting_param_values()

    # Are there different active classes per context (or just potentially a different mask per context)?
    per_context = (model.scenario=="task" or (model.scenario=="class" and model.neg_samples=="current"))
    per_context_singlehead = per_context and (model.scenario=="task" and model.singlehead)
    if model.experiment=="CIFAR50" or model.experiment=="MINI":
        first_classes = 50
    elif model.experiment=="TINY":
        first_classes = 100
    else:
        first_classes = 10

    # Loop over all contexts.
    for context, train_dataset in enumerate(train_datasets, 1):

        # If using the "joint" baseline, skip to last context, as model is only be trained once on data of all contexts
        if baseline=='joint':
            if context<len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        # If using the "cummulative" (or "joint") baseline, create a large training dataset of all contexts so far
        if baseline=="cummulative" and (not per_context):
            train_dataset = ConcatDataset(train_datasets[:context])
        # -but if "cummulative"+[per_context]: training on each context must be separate, as a trick to achieve this,
        #                                      all contexts so far are treated as replay (& there is no current batch)
        if baseline=="cummulative" and per_context:
            ReplayStoredData = True
            previous_datasets = train_datasets

        # Add memory buffer (if available) to current dataset (if requested)
        if checkattr(model, 'add_buffer') and context>1:
            if model.scenario=="domain" or per_context_singlehead:
                target_transform = (lambda y, x=model.classes_per_context: y % x)
            else:
                target_transform = None
            memory_dataset = MemorySetDataset(model.memory_sets, target_transform=target_transform)
            training_dataset = ConcatDataset([train_dataset, memory_dataset])
        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update (needed for SI)
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
            W, p_old = model.prepare_importance_estimates_dicts()

        # Find [active_classes]
        if model.scenario=="task":
            if not model.singlehead:
                # -for Task-IL scenario, create <list> with for all contexts so far a <list> with the active classes
                active_classes = [list(
                    range(model.classes_per_context * i, model.classes_per_context * (i+1))
                ) for i in range(context)]
            else:
                #--> if a single-headed output layer is used in the Task-IL scenario, all output units are always active
                active_classes = None
        elif model.scenario=="domain":
            # -for Domain-IL scenario, always all classes are active
            active_classes = None
        elif model.scenario=="class":
            # -for Class-IL scenario, the active classes are determined by [model.neg_samples]
            if model.neg_samples=="all-so-far":
                # --> one <list> with active classes of all contexts so far
                active_classes = list(range(first_classes + model.classes_per_context * (context-1)))
                # if model.experiment!="CIFAR50" and model.experiment!='MINI' and model.experiment!='TINY':
                #     active_classes = list(range(model.classes_per_context * context))
                # elif context==1:
                #     active_classes = list(range(50))
                # else:
                #     active_classes = list(range(50 + model.classes_per_context * (context-1)))
            elif model.neg_samples=="all":
                #--> always all classes are active
                active_classes = None
            elif model.neg_samples=="current":
                #--> only those classes in the current or replayed context are active (i.e., train "as if Task-IL")
                if context==1:
                    active_classes = [list(
                        range(first_classes)
                    ) ]
                else:
                    active_classes = [list(range(50))] + [list(
                        range(first_classes+model.classes_per_context * i, first_classes+model.classes_per_context * (i + 1))
                    ) for i in range(context-1)]
                # if model.experiment!="CIFAR50" and model.experiment!='MINI':
                #     active_classes = [list(
                #         range(model.classes_per_context * i, model.classes_per_context * (i + 1))
                #     ) for i in range(context)]
                # else:
                #     if context==1:
                #         active_classes = [list(
                #         range(50)
                #     ) ]
                #     else:
                #         active_classes = [list(range(50))] + [list(
                #         range(50+model.classes_per_context * i, 50+model.classes_per_context * (i + 1))
                    # ) for i in range(context-1)]

        # Reset state of optimizer(s) for every context (if requested)
        if (not model.label=="SeparateClassifiers") and model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if per_context:
            up_to_context = context if baseline=="cummulative" else context-1
            iters_left_previous = [1]*up_to_context
            data_loader_previous = [None]*up_to_context

        # Define tqdm progress bar(s)
        if context==1:
            progress = tqdm.tqdm(range(1, first_iters+1))
        else:
            progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))

        # Loop over all iterations
        if context==1:
            iters_to_use = first_iters
        else:
            iters_to_use = iters if (generator is None) else max(iters, gen_iters)

        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                # NOTE:  [train_dataset]  is training-set of current context
                #      [training_dataset] is training-set of current context with stored samples added (if requested)
                iters_left = len(data_loader)
            if ReplayStoredData:
                if per_context:
                    up_to_context = context if baseline=="cummulative" else context-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_context)) if (up_to_context>1) else batch_size
                    # -if different active classes per context (e.g., Task-IL), need separate replay for each context
                    for context_id in range(up_to_context):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[context_id]))
                        iters_left_previous[context_id] -= 1
                        if iters_left_previous[context_id]==0:
                            data_loader_previous[context_id] = iter(get_data_loader(
                                previous_datasets[context_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[context_id] = len(data_loader_previous[context_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(get_data_loader(ConcatDataset(previous_datasets),
                                                                    batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            if baseline=="cummulative" and per_context:
                x = y = scores = None
            else:
                x, y = next(data_loader)                             #--> sample training data of current context
                y = y-model.classes_per_context*(context-1) if per_context and not per_context_singlehead else y
                # --> adjust the y-targets to the 'active range'
                x, y = x.to(device), y.to(device)                    #--> transfer them to correct device
                # If --bce & --bce-distill, calculate scores for past classes of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and model.scenario in ("class", "all") and (previous_model is not None):
                    with torch.no_grad():
                        if model.experiment!="CIFAR50" and model.experiment!='MINI' and model.experiment!='TINY':
                            scores = previous_model.classify(
                                x, no_prototypes=True
                            )[:, :(model.classes_per_context * (context - 1))]
                        else:
                            if context==1:
                                scores = previous_model.classify(
                                x, no_prototypes=True
                            )[:, :(0)]
                            else:
                                scores = previous_model.classify(
                                x, no_prototypes=True
                            )[:, :(first_classes+model.classes_per_context * (context - 2))]
                else:
                    scores = None


            #####-----REPLAYED BATCH-----#####
            if not ReplayStoredData and not ReplayGeneratedData and not ReplayCurrentData:
                x_ = y_ = scores_ = context_used = mu_dist = logvar_dist = None   #-> if no replay
                gen_data = []
            ##-->> Replay of stored data <<--##
            if ReplayStoredData:
                scores_ = context_used = None
                if not per_context:
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets=="hard") else None
                    # If required, get target scores (i.e, [scores_])         -- using previous model, with no_grad()
                    if (model.replay_targets=="soft"):
                        with torch.no_grad():
                            scores_ = previous_model.classify(x_, no_prototypes=True)
                        if model.scenario=="class" and model.neg_samples=="all-so-far":
                            scores_ = scores_[:, :(first_classes + model.classes_per_context*(context-2))]
                            # if model.experiment!="CIFAR50" and model.experiment!='MINI':
                            #     scores_ = scores_[:, :(model.classes_per_context*(context-1))]
                            # else:
                            #     scores_ = scores_[:, :(50 + model.classes_per_context*(context-2))]
                            #-> if [scores_] is not same length as [x_], zero probs are added in [loss_fn_kd]-function
                else:
                    # Sample replayed training data, move to correct device and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_context = context if baseline=="cummulative" else context-1
                    for context_id in range(up_to_context):
                        x_temp, y_temp = next(data_loader_previous[context_id])
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets=="hard":
                            if not per_context_singlehead:
                                y_temp = y_temp - (model.classes_per_context*context_id) #-> adjust y to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    # If required, get target scores (i.e, [scores_])        -- using previous model, with no_grad()
                    if (model.replay_targets=="soft") and (previous_model is not None):
                        scores_ = list()
                        for context_id in range(up_to_context):
                            with torch.no_grad():
                                scores_temp = previous_model.classify(x_[context_id], no_prototypes=True)
                            if active_classes is not None:
                                scores_temp = scores_temp[:, active_classes[context_id]]
                            scores_.append(scores_temp)

            ##-->> Generative / Current Replay <<--##

            #---INPUTS---#
            if ReplayCurrentData:
                x_ = x  #--> use current context inputs
                context_used = None

            if ReplayGeneratedData:
                conditional_gen = True if previous_generator.label=='CondVAE' and \
                                          ((previous_generator.per_class and previous_generator.prior=="GMM")
                                           or checkattr(previous_generator, 'dg_gates')) else False
                if conditional_gen and per_context:
                    # -if a cond generator is used with different active classes per context, generate data per context
                    x_ = list()
                    context_used = list()
                    for context_id in range(context-1):
                        allowed_domains = list(range(context - 1))
                        allowed_classes = list(
                            range(model.classes_per_context*context_id, model.classes_per_context*(context_id+1))
                        )
                        batch_size_to_use = int(np.ceil(batch_size / (context-1)))
                        x_temp_ = previous_generator.sample(batch_size_to_use, allowed_domains=allowed_domains,
                                                            allowed_classes=allowed_classes, only_x=False)
                        x_.append(x_temp_[0])
                        context_used.append(x_temp_[2])
                else:
                    # -which classes are allowed to be generated? (relevant if conditional generator / decoder-gates)
                    allowed_classes = None if model.scenario=="domain" else list(
                        range(first_classes + model.classes_per_context*(context-2))
                    )
                    # if model.experiment=="CIFAR50" and model.experiment!='MINI':
                    #     allowed_classes = None if model.scenario=="domain" else list(
                    #         range(50 + model.classes_per_context*(context-2))
                    #     )            
                    # else:
                    #     allowed_classes = None if model.scenario=="domain" else list(
                    #         range(model.classes_per_context*(context-1))
                    #     )                 
                    # -which contexts are allowed to be generated? (only relevant if "Domain-IL" with context-gates)
                    allowed_domains = list(range(context-1))
                    # -generate inputs representative of previous contexts
                    x_temp_ = previous_generator.sample(batch_size, allowed_classes=allowed_classes,
                                                        allowed_domains=allowed_domains, only_x=False)
                    x_ = x_temp_[0] if type(x_temp_)==tuple else x_temp_
                    y_temp_cycle_ = x_temp_[1]
                    for cycle in range(cycles):
                        x_ = previous_generator(x_, gate_input=y_temp_cycle_, full=False)
                    context_used = x_temp_[2] if type(x_temp_)==tuple else None

            #---OUTPUTS---#
            if ReplayGeneratedData or ReplayCurrentData:
                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                if not per_context:
                    # -if replay does not need to be evaluated separately for each context
                    with torch.no_grad():
                        # logging.info("heloooooooooooooooooooooooooooo")
                        scores_ = previous_model.classify(x_, no_prototypes=True)
                        _, label = torch.max(scores_, dim=1)
                        _,_,_,mu_dist,logvar_dist,_,_,_ = previous_model.forward(x_, full=True, gate_input = label)
                    if model.scenario == "class" and model.neg_samples == "all-so-far":
                        scores_ = scores_[:, :(first_classes+model.classes_per_context * (context - 2))]
                        # if model.experiment=="CIFAR50" and model.experiment!='MINI':

                        #     scores_ = scores_[:, :(50+model.classes_per_context * (context - 2))]
                        # else:
                        #     scores_ = scores_[:, :(model.classes_per_context * (context - 1))]
                        # -> if [scores_] is not same length as [x_], zero probs are added in [loss_fn_kd]-function
                    # -also get the 'hard target'
                    _, y_ = torch.max(scores_, dim=1)
                else:
                    # -[x_] needs to be evaluated according to each past context, so make list with entry per context
                    scores_ = list()
                    y_ = list()
                    # -if no context-mask and no conditional generator, all scores can be calculated in one go
                    if previous_model.mask_dict is None and not type(x_)==list:
                        with torch.no_grad():
                            all_scores_ = previous_model.classify(x_, no_prototypes=True)
                            _,_,_,mu_dist,logvar_dist,_,_,_ = previous_model.forward(x_, full=True)
                    for context_id in range(context-1):
                        # -if there is a context-mask (i.e., XdG), obtain predicted scores for each context separately
                        if previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(context=context_id+1)
                        if previous_model.mask_dict is not None or type(x_)==list:
                            with torch.no_grad():
                                all_scores_ = previous_model.classify(x_[context_id] if type(x_)==list else x_,
                                                                      no_prototypes=True)
                        temp_scores_ = all_scores_
                        if active_classes is not None:
                            temp_scores_ = temp_scores_[:, active_classes[context_id]]
                        scores_.append(temp_scores_)
                        # - also get hard target
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        y_.append(temp_y_)

                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None


            #---> Train MAIN MODEL
            if batch_index <= iters_to_use:
                # if previous_model is not None:
                    
                #     vals = [1000,3000,4000]
                #     if batch_index in vals:
                #         num_iters-=1
                #     for it in range(num_iters):
                #         x = previous_model(x, gate_input=y)
                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_, mu_dist=mu_dist,logvar_dist = logvar_dist,rnt = 1./context,
                                                contexts_=context_used, active_classes=active_classes, context=context)

                # Update running parameter importance estimates in W (needed for SI)
                if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
                    model.update_importance_estimates(W, p_old)

                # Fire callbacks (for visualization of training-progress / evaluating performance after each context)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, context=context)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, context=context)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, context=context)


            #---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, x_=x_, rnt=1./context)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, context=context)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, context=context)


        ##----------> UPON FINISHING EACH CONTEXT...

        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()

        # Parameter regularization: update and compute the parameter importance estimates
        if context<len(train_datasets) and isinstance(model, ContinualLearner):
            # -find allowed classes
            allowed_classes = active_classes[-1] if (per_context and not per_context_singlehead) else active_classes
            # -if needed, apply correct context-specific mask
            if model.mask_dict is not None:
                model.apply_XdGmask(context=context)
            ##--> EWC/NCL: estimate the Fisher Information matrix
            if model.importance_weighting=='fisher' and (model.weight_penalty or model.precondition):
                if model.fisher_kfac:
                    model.estimate_kfac_fisher(training_dataset, allowed_classes=allowed_classes)
                else:
                    model.estimate_fisher(training_dataset, allowed_classes=allowed_classes)
            ##--> OWM: calculate and update the projection matrix
            if model.importance_weighting=='owm' and (model.weight_penalty or model.precondition):
                model.estimate_owm_fisher(training_dataset, allowed_classes=allowed_classes)
            ##--> SI: calculate and update the normalized path integral
            if model.importance_weighting=='si' and (model.weight_penalty or model.precondition):
                model.update_omega(W, model.epsilon)

        # MEMORY BUFFER: update the memory buffer
        if checkattr(model, 'use_memory_buffer'):
            samples_per_class = model.budget_per_class if (not model.use_full_capacity) else int(
                np.floor((model.budget_per_class*len(train_datasets))/context)
            )
            # reduce examplar-sets (only needed when '--use-full-capacity' is selected)
            model.reduce_memory_sets(samples_per_class)
            # for each new class trained on, construct examplar-set
            if context==1:
                new_classes = list(range(first_classes)) if (
                    model.scenario=="domain" or per_context_singlehead
            ) else list(range(0, first_classes))
            else:
                new_classes = list(range(model.classes_per_context)) if (
                        model.scenario=="domain" or per_context_singlehead
                ) else list(range(first_classes+model.classes_per_context*(context-2), 50+model.classes_per_context*(context-1)))
            # if model.experiment!="CIFAR50" and model.experiment!='MINI':
            #     new_classes = list(range(model.classes_per_context)) if (
            #             model.scenario=="domain" or per_context_singlehead
            #     ) else list(range(model.classes_per_context*(context-1), model.classes_per_context*context))
            # else:
            #     if context==1:
            #         new_classes = list(range(50)) if (
            #             model.scenario=="domain" or per_context_singlehead
            #     ) else list(range(0, 50))
            #     else:
            #         new_classes = list(range(model.classes_per_context)) if (
            #                 model.scenario=="domain" or per_context_singlehead
            #         ) else list(range(50+model.classes_per_context*(context-2), 50+model.classes_per_context*(context-1)))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new memory-set for this class
                allowed_classes = active_classes[-1] if per_context and not per_context_singlehead else active_classes
                model.construct_memory_set(dataset=class_dataset, n=samples_per_class, label_set=allowed_classes)
            model.compute_means = True

        # Run the callbacks after finishing each context
        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters_to_use, context=context)
        # if context>1:
        #     for it in range(100):
        #         # print(dataset)
        #         # data_loader = get_data_loader(dataset, len(dataset), cuda=cuda, shuffle=False)
        #         # z_class = []
        #         model.eval()
        #         with torch.no_grad():
        #             # for x,y in data_loader:
        #             #     _, _, _, _, z = model.forward(x.to(device), gate_input=y.to(device), full=True)
        #             #     z_class.append(z)
        #             # z_class = torch.cat(z_class, dim=0)
        #             # mean_class, var_class = torch.var_mean(z_class, dim=0)
        #             reconL = F.mse_loss(input=model.z_class_means[it], target=previous_model.z_class_means[it], reduction='none')
        #             reconL = torch.mean(reconL).item()
        #             # reconL = -lf.log_Normal_standard(x=mean_class, mean=model.z_class_means[y[0]], average=True, dim=-1)
        #             # reconL = lf.weighted_average(reconL).item()
        #             # logging.info(model.z_class_means[y[0]])
        #             logging.info(np.sqrt(reconL))
        #         model.train()
        # REPLAY: update source for replay
        if context<len(train_datasets) and hasattr(model, 'replay_mode'):
            previous_model = copy.deepcopy(model).eval()
            if model.replay_mode == 'generative':
                ReplayGeneratedData = True
                previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
            elif model.replay_mode == 'current':
                ReplayCurrentData = True
            elif model.replay_mode in ('buffer', 'all'):
                ReplayStoredData = True
                if model.replay_mode == "all":
                    previous_datasets = train_datasets[:context]
                else:
                    if per_context:
                        previous_datasets = []
                        for context_id in range(context):
                            if context_id==0:
                                previous_datasets.append(MemorySetDataset(
                                model.memory_sets[
                                    0:first_classes
                                ],
                                target_transform=(lambda y, x=0: y + x) if (
                                    not per_context_singlehead
                                ) else (lambda y, x=first_classes: y % x)
                            ))
                            else:
                                previous_datasets.append(MemorySetDataset(
                                model.memory_sets[
                                    (first_classes+model.classes_per_context * (context_id-1)):(first_classes+model.classes_per_context*(context_id))
                                ],
                                target_transform=(lambda y, x=first_classes+model.classes_per_context * (context_id-1): y + x) if (
                                    not per_context_singlehead
                                ) else (lambda y, x=model.classes_per_context: y % x)
                            ))
                            # if model.experiment!="CIFAR50" and model.experiment!='MINI':
                            #     previous_datasets.append(MemorySetDataset(
                            #         model.memory_sets[
                            #             (model.classes_per_context * context_id):(model.classes_per_context*(context_id+1))
                            #         ],
                            #         target_transform=(lambda y, x=model.classes_per_context * context_id: y + x) if (
                            #             not per_context_singlehead
                            #         ) else (lambda y, x=model.classes_per_context: y % x)
                            #     ))
                            # else:
                            #     if context_id==0:
                            #         previous_datasets.append(MemorySetDataset(
                            #         model.memory_sets[
                            #             0:50
                            #         ],
                            #         target_transform=(lambda y, x=0: y + x) if (
                            #             not per_context_singlehead
                            #         ) else (lambda y, x=50: y % x)
                            #     ))
                            #     else:
                            #         previous_datasets.append(MemorySetDataset(
                            #         model.memory_sets[
                            #             (50+model.classes_per_context * (context_id-1)):(50+model.classes_per_context*(context_id))
                            #         ],
                            #         target_transform=(lambda y, x=50+model.classes_per_context * (context_id-1): y + x) if (
                            #             not per_context_singlehead
                            #         ) else (lambda y, x=model.classes_per_context: y % x)
                            #     ))
                    else:
                        if context==1:
                            target_transform = None if not model.scenario=="domain" else (
                            lambda y, x=first_classes: y % x
                        )
                        else:
                            target_transform = None if not model.scenario=="domain" else (
                            lambda y, x=model.classes_per_context: y % x
                        )
                        # if model.experiment!="CIFAR50" and model.experiment!='MINI':
                        #     target_transform = None if not model.scenario=="domain" else (
                        #         lambda y, x=model.classes_per_context: y % x
                        #     )
                        # else:
                        #     if context==1:
                        #         target_transform = None if not model.scenario=="domain" else (
                        #         lambda y, x=50: y % x
                        #     )
                        #     else:
                        #         target_transform = None if not model.scenario=="domain" else (
                        #         lambda y, x=model.classes_per_context: y % x
                        #     )
                        previous_datasets = [MemorySetDataset(model.memory_sets, target_transform=target_transform)]

        progress.close()
        if generator is not None:
            progress_gen.close()

        # accs = []
        # for i in range(context):
        #     acc = evaluate.test_acc(
        #         model, test_datasets[i], verbose=False, test_size=None, context_id=i, allowed_classes=list(
        #             range(0, config['classes_per_context']*(i+1))
        #         )
        #     )
        #     accs.append(acc)
        #     print(" - Context {}: {:.4f}".format(i + 1, acc))
        # average_accs = sum(accs) / (context)
        # print('=> average accuracy over all {} contexts: {:.4f}\n\n'.format(context, average_accs))

        accs = []
        
        # rec_losses = []
        # for i in range(context):
        #     if len(gen_data)<i+1:                    
        #         acc, gen, rec_loss = evaluate.test_acc(
        #             model, test_datasets[i], gen_data=None,verbose=False, test_size=None, context_id=i, allowed_classes=None
        #         )
        #         gen_data.append(gen)
        #     else:
        #         acc, gen, rec_loss = evaluate.test_acc(
        #             model, test_datasets[i], gen_data=gen_data[i],verbose=False, test_size=None, context_id=i, allowed_classes=None
        #         )
        #         gen_data[i] = gen
        #     rec_losses.append(rec_loss)
        #     accs.append(acc)
        #     print(" - Context {}: {:.4f}".format(i + 1, acc))
        #     print(f"Reconstruction loss for context {i+1}: {rec_loss}")
        # average_accs = sum(accs) / (context)
        # print('=> average accuracy over all {} contexts: {:.4f}\n\n'.format(context, average_accs))
        
        # average_rec_loss = sum(rec_losses)/context
        # print(f"=> average rec_loss over all {context} contexts: {average_rec_loss}")

        for i in range(context):
            if len(gen_data)<i+1:                    
                acc = evaluate.test_acc(
                    model, test_datasets[i], gen_data=None,verbose=False, test_size=None, context_id=i, allowed_classes=None
                )
                # gen_data.append(gen)
            else:
                acc = evaluate.test_acc(
                    model, test_datasets[i], gen_data=None,verbose=False, test_size=None, context_id=i, allowed_classes=None
                )
                # gen_data[i] = gen
            # rec_losses.append(rec_loss)
            accs.append(acc)
            logging.info(" - Context {}: {:.4f}".format(i + 1, acc))
            # print(f"Reconstruction loss for context {i+1}: {rec_loss}")
        average_accs = sum(accs) / (context)
        logging.info('=> average accuracy over all {} contexts: {:.4f}\n\n'.format(context, average_accs))
        
        concat_dataset = ConcatDataset([test_datasets[i] for i in range(context)])
        # gen_size = 0
        # for i in range(context):
        #     gen_size += len(test_datasets[i])
        gen_size = len(concat_dataset)
        # test_datasets[i]
        allowed_domains = list(range(context))
        # generations = model.sample(gen_size, allowed_classes=active_classes,
        #                                                 allowed_domains=allowed_domains, only_x=False)
        x_temp_ = model.sample(gen_size, allowed_classes=active_classes,
                                    allowed_domains=allowed_domains, only_x=False)
        generations = x_temp_[0] if type(x_temp_)==tuple else x_temp_
        y_temp_cycle_ = x_temp_[1]
        for cycle in range(cycles):
            generations = model(generations, gate_input=y_temp_cycle_, full=False)
        # y_temp_cycle_ = x_temp_[1]
        # for cycle in range(cycles):
        #     generations = model(generations, gate_input=y_temp_cycle_, full=False)
        # _,_,generations,_ = model.encode(generations)
        n_repeats = int(np.ceil(gen_size/batch_size))
        gen_emb  = []
        for i in range(n_repeats):
            x = generations[(i*batch_size): int(min(((i+1)*batch_size), gen_size))]
            with torch.no_grad():
                gen_emb.append(x.cpu().numpy())
        gen_emb  = np.concatenate(gen_emb)
        data_loader = get_data_loader(concat_dataset, batch_size=batch_size, cuda=cuda)
        real_emb = []
        for real_x, _ in data_loader:
            with torch.no_grad():
                _,_,real_x,_ = model.encode(real_x.cuda())
                real_emb.append(real_x.cpu().numpy())
        real_emb = np.concatenate(real_emb)
        precision, recall = pr.compute_prd_from_embedding(gen_emb, real_emb)
        logging.info(f'precision: {precision}, recall: {recall}')
        figure = plot_pr_curves([[precision]], [[recall]])
        figure.savefig(f"/raid/NFS_SHARE/home/valeriya.khan/continual-learning/logs/figs/recall_prec_{context}_{seed}_latent_cycles_fix.png")
        utils.save_checkpoint(model, '/raid/NFS_SHARE/home/valeriya.khan/continual-learning/store/models/laten_match_dist_cycle/', name=f'model-{model.experiment}-seed{seed}-cycles{cycles}-context{context}')
        # pp.savefig(figure)
        # average_rec_loss = sum(rec_losses)/context
        # print(f"=> average rec_loss over all {context} contexts: {average_rec_loss}")

        # if model.label == "VAE" or model.label == "CondVAE":
        #     rec_losses = []
        #     for i in range(context):
        #         rec_loss = evaluate.test_degradation(
        #             model, test_datasets[i], verbose=False, test_size=None, context_id=i, allowed_classes=None
        #         )
        #         rec_losses.append(rec_loss)
        #         print(" - Context {}: {:.4f}".format(i + 1, rec_loss))
        #     average_accs = sum(rec_losses) / (context)
        #     print('=> reconstruction loss over all {} contexts: {:.4f}\n\n'.format(context, average_accs))

#------------------------------------------------------------------------------------------------------------#

def train_fromp(model, train_datasets,test_datasets, config, iters=2000, batch_size=32,
                loss_cbs=list(), eval_cbs=list(), context_cbs=list(), first_iters = 0,**kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts using the FROMP algorithm.

    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Are there different active classes per context (or just potentially a different mask per context)?
    per_context = (model.scenario=="task" or (model.scenario=="class" and model.neg_samples=="current"))
    per_context_singlehead = per_context and (model.scenario=="task" and model.singlehead)

    # Loop over all contexts.
    for context, train_dataset in enumerate(train_datasets, 1):

        # Find [active_classes]
        if model.scenario=="task":
            if not model.singlehead:
                # -for Task-IL scenario, create <list> with for all contexts so far a <list> with the active classes
                active_classes = [list(
                    range(model.classes_per_context * i, model.classes_per_context * (i+1))
                ) for i in range(context)]
            else:
                #--> if a single-headed output layer is used in the Task-IL scenario, all output units are always active
                active_classes = None
        elif model.scenario=="domain":
            # -for Domain-IL scenario, always all classes are active
            active_classes = None
        elif model.scenario=="class":
            # -for Class-IL scenario, the active classes are determined by [model.neg_samples]
            if model.neg_samples=="all-so-far":
                # --> one <list> with active classes of all contexts so far
                active_classes = list(range(model.classes_per_context * context))
            elif model.neg_samples=="all":
                #--> always all classes are active
                active_classes = None
            elif model.neg_samples=="current":
                #--> only those classes in the current or replayed context are active (i.e., train "as if Task-IL")
                active_classes = [list(
                    range(model.classes_per_context * i, model.classes_per_context * (i + 1))
                ) for i in range(context)]

        # Find [label_sets] (i.e., when replaying/revisiting/regularizing previous contexts, which labels to consider)
        label_sets = active_classes if (per_context and not per_context_singlehead) else [active_classes]*context
        # NOTE: With Class-IL, when revisiting previous contexts, consider all labels up to *now*
        #       (and not up to when that context was encountered!)

        # FROMP: calculate and store regularisation-term-related quantities
        if context > 1:
            model.optimizer.init_context(context-1, reset=(model.optim_type=="adam_reset"),
                                         classes_per_context=model.classes_per_context, label_sets=label_sets)

        # Initialize # iters left on current data-loader(s)
        iters_left = 1

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        for batch_index in range(1, iters+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=True))
                iters_left = len(data_loader)

            # -----------------Collect data------------------#
            x, y = next(data_loader)           #--> sample training data of current context
            y = y - model.classes_per_context * (context - 1) if (per_context and not per_context_singlehead) else y
            # --> adjust the y-targets to the 'active range'
            x, y = x.to(device), y.to(device)  # --> transfer them to correct device

            #---> Train MAIN MODEL
            if batch_index <= iters:

                # Optimiser step
                loss_dict = model.optimizer.step(x, y, label_sets, context-1, model.classes_per_context)

                # Fire callbacks (for visualization of training-progress / evaluating performance after each context)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, context=context)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, context=context)

        ##----------> UPON FINISHING EACH CONTEXT...

        # Close progres-bar(s)
        progress.close()

        # MEMORY BUFFER: update the memory buffer
        if checkattr(model, 'use_memory_buffer'):
            samples_per_class = model.budget_per_class if (not model.use_full_capacity) else int(
                np.floor((model.budget_per_class*len(train_datasets))/context)
            )
            # reduce examplar-sets (only needed when '--use-full-capacity' is selected)
            model.reduce_memory_sets(samples_per_class)
            # for each new class trained on, construct examplar-set
            new_classes = list(range(model.classes_per_context)) if (
                    model.scenario=="domain" or per_context_singlehead
            ) else list(range(model.classes_per_context*(context-1), model.classes_per_context*context))
            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, sub_labels=[class_id])
                # based on this dataset, construct new memory-set for this class
                allowed_classes = active_classes[-1] if per_context and not per_context_singlehead else active_classes
                model.construct_memory_set(dataset=class_dataset, n=samples_per_class, label_set=allowed_classes)
            model.compute_means = True

        # FROMP: update covariance (\Sigma)
        if context<len(train_datasets):
            memorable_loader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=False, num_workers=3)
            model.optimizer.update_fisher(
                memorable_loader,
                label_set=active_classes[context-1] if (per_context and not per_context_singlehead) else active_classes
            )

        # Run the callbacks after finishing each context
        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters, context=context)

#------------------------------------------------------------------------------------------------------------#

def train_gen_classifier(model, train_datasets, test_datasets, config, iters=2000, epochs=None, batch_size=32,first_iters=0,
                         loss_cbs=list(), sample_cbs=list(), eval_cbs=list(), context_cbs=list(), **kwargs):
    '''Train a generative classifier with a separate VAE per class.

    [model]               <nn.Module> the generative classifier to train
    [train_datasets]      <list> with for each class the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per class
    [batch_size]          <int>, # of samples per mini-batch
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Use cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Loop over all contexts.
    classes_in_current_context = 0
    context = 1
    for class_id, train_dataset in enumerate(train_datasets):

        # Initialize # iters left on data-loader(s)
        iters_left = 1

        if epochs is not None:
            data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda, drop_last=False))
            iters = len(data_loader)*epochs

        # Define a tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        # Loop over all iterations
        for batch_index in range(1, iters+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(get_data_loader(train_dataset, batch_size, cuda=cuda,
                                                   drop_last=True if epochs is None else False))
                iters_left = len(data_loader)

            # Collect data
            x, y = next(data_loader)                                    #--> sample training data of current context
            x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
            #y = y.expand(1) if len(y.size())==1 else y                 #--> hack for if batch-size is 1

            # Select model to be trained
            model_to_be_trained = getattr(model, "vae{}".format(class_id))

            # Train the VAE model of this class with this batch
            loss_dict = model_to_be_trained.train_a_batch(x)

            # Fire callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, class_id=class_id)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index+classes_in_current_context*iters, context=context)
            for sample_cb in sample_cbs:
                if sample_cb is not None:
                    sample_cb(model_to_be_trained, batch_index, class_id=class_id)

        # Close progres-bar(s)
        progress.close()

        # Did a context just finish?
        classes_in_current_context += 1
        if classes_in_current_context==model.classes_per_context:
            # Run the callbacks after finishing each context
            for context_cb in context_cbs:
                if context_cb is not None:
                    context_cb(model, iters, context=context)
            # Updated counts
            classes_in_current_context = 0
            context += 1

#------------------------------------------------------------------------------------------------------------#

def train_on_stream(model, datastream, iters=2000, loss_cbs=list(), eval_cbs=list()):
    '''Incrementally train a model on a ('task-free') stream of data.
    Args:
        model (Classifier): model to be trained, must have a built-in `train_a_batch`-method
        datastream (DataStream): iterator-object that returns for each iteration the training data
        iters (int, optional): max number of iterations, could be smaller if `datastream` runs out (default: ``2000``)
        *_cbs (list of callback-functions, optional): for evaluating training-progress (defaults: empty lists)
    '''

    # Define tqdm progress bar(s)
    progress = tqdm.tqdm(range(1, iters + 1))

    ##--> SI: Register starting parameter values
    if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
        start_new_W = True
        model.register_starting_param_values()

    previous_model = None

    for batch_id, (x,y,c) in enumerate(datastream, 1):

        if batch_id > iters:
            break

        ##--> SI: Prepare <dicts> to store running importance estimates and param-values before update
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
            if start_new_W:
                W, p_old = model.prepare_importance_estimates_dicts()
                start_new_W = False

        # Move data to correct device
        x = x.to(model._device())
        y = y.to(model._device())
        if c is not None:
            c = c.to(model._device())

        # If using separate networks, the y-targets need to be adjusted
        if model.label == "SeparateClassifiers":
            for sample_id in range(x.shape[0]):
                y[sample_id] = y[sample_id] - model.classes_per_context * c[sample_id]

        # Add replay...
        (x_, y_, c_, scores_) = (None, None, None, None)
        if hasattr(model, 'replay_mode') and model.replay_mode=='buffer' and previous_model is not None:
            # ... from the memory buffer
            (x_, y_, c_) = previous_model.sample_from_buffer(x.shape[0])
            if model.replay_targets=='soft':
                with torch.no_grad():
                    scores_ = previous_model.classify(x_, c_, no_prototypes=True)
        elif hasattr(model, 'replay_mode') and model.replay_mode=='current' and previous_model is not None:
            # ... using the data from the current batch (as in LwF)
            x_ = x
            if c is not None:
                c_ = previous_model.sample_contexts(x_.shape[0]).to(model._device())
            with torch.no_grad():
                scores_ = previous_model.classify(x, c_, no_prototypes=True)
                _, y_ = torch.max(scores_, dim=1)
        # -only keep [y_] or [scores_], depending on whether replay is with 'hard' or 'soft' targets
        y_ = y_ if (hasattr(model, 'replay_targets') and model.replay_targets == "hard") else None
        scores_ = scores_ if (hasattr(model, 'replay_targets') and model.replay_targets == "soft") else None

        # Train the model on this batch
        loss_dict = model.train_a_batch(x, y, c, x_=x_, y_=y_, c_=c_, scores_=scores_, rnt=0.5)

        ##--> SI: Update running parameter importance estimates in W (needed for SI)
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si':
            model.update_importance_estimates(W, p_old)

        # Add the observed data to the memory buffer (if selected by the algorithm that fills the memory buffer)
        if checkattr(model, 'use_memory_buffer'):
            model.add_new_samples(x, y, c)
        if hasattr(model, 'replay_mode') and model.replay_mode == 'current' and c is not None:
            model.keep_track_of_contexts_so_far(c)

        # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
        for loss_cb in loss_cbs:
            if loss_cb is not None:
                loss_cb(progress, batch_id, loss_dict)
        for eval_cb in eval_cbs:
            if eval_cb is not None:
                eval_cb(model, batch_id, context=None)

        ##--> SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and model.importance_weighting=='si' and model.weight_penalty:
            if (batch_id % model.update_every)==0:
                model.update_omega(W, model.epsilon)
                start_new_W = True

        ##--> Replay: update source for replay
        if hasattr(model, 'replay_mode') and (not model.replay_mode=="none"):
            if (batch_id % model.update_every)==0:
                previous_model = copy.deepcopy(model).eval()

    # Close progres-bar(s)
    progress.close()

#------------------------------------------------------------------------------------------------------------#

def train_gen_classifier_on_stream(model, datastream, iters=2000, loss_cbs=list(), eval_cbs=list()):
    '''Incrementally train a generative classifier model on a ('task-free') stream of data.
    Args:
        model (Classifier): generative classifier, each generative model must have a built-in `train_a_batch`-method
        datastream (DataStream): iterator-object that returns for each iteration the training data
        iters (int, optional): max number of iterations, could be smaller if `datastream` runs out (default: ``2000``)
        *_cbs (list of callback-functions, optional): for evaluating training-progress (defaults: empty lists)
    '''

    # Define tqdm progress bar(s)
    progress = tqdm.tqdm(range(1, iters + 1))

    for batch_id, (x,y,_) in enumerate(datastream, 1):

        if batch_id > iters:
            break

        # Move data to correct device
        x = x.to(model._device())
        y = y.to(model._device())

        # Cycle through all classes. For each class present, take training step on corresponding generative model
        for class_id in range(model.classes):
            if class_id in y:
                x_to_use = x[y==class_id]
                loss_dict = getattr(model, "vae{}".format(class_id)).train_a_batch(x_to_use)
                # NOTE: this way, only the [lost_dict] of the last class present in the batch enters into the [loss_cb]

        # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
        for loss_cb in loss_cbs:
            if loss_cb is not None:
                loss_cb(progress, batch_id, loss_dict)
        for eval_cb in eval_cbs:
            if eval_cb is not None:
                eval_cb(model, batch_id, context=None)

    # Close progres-bar(s)
    progress.close()