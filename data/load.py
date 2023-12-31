import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.manipulate import permutate_image_pixels, SubDataset, TransformedDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
import logging
from PIL import Image



def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'MNIST' if name in ('MNIST28', 'MNIST32') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    if data_name=='MINI':
        transforms_list = [*AVAILABLE_TRANSFORMS['augment_mini']] if augment else []
    elif data_name=='TINY':
        transforms_list = [*AVAILABLE_TRANSFORMS['augment_tiny']]
    elif data_name=='IN100':
        transforms_list = [*AVAILABLE_TRANSFORMS['augment_IN100']] 
    else:
        transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    if data_name=='TINY':
        dataset = dataset_class(f'{dir}/{data_name}/tiny-imagenet-200/{type}', transform=dataset_transform, target_transform=target_transform)
    elif data_name=='IN100':
        class_order = ['n03710193', 'n03089624', 'n04152593', 'n01806567', 'n02107574', 'n04409515', 'n04599235', 'n03657121', 'n03942813', 'n04026417',
                      'n02640242', 'n04591157', 'n01689811', 'n07614500', 'n03085013', 'n01882714', 'n02112706', 'n04266014', 'n02786058', 'n02526121',
                      'n03141823', 'n03775071', 'n04074963', 'n01531178', 'n04428191', 'n02096177', 'n02091467', 'n02971356', 'n02116738', 'n03017168',
                      'n02002556', 'n04355933', 'n02840245', 'n04371430', 'n01774384', 'n03223299', 'n04399382', 'n02088094', 'n02033041', 'n02814860',
                      'n04604644', 'n02669723', 'n03884397', 'n03250847', 'n04153751', 'n03016953', 'n02101388', 'n01914609', 'n02128385', 'n03075370',
                      'n02363005', 'n09468604', 'n02011460', 'n03785016', 'n12267677', 'n12768682', 'n12620546', 'n01537544', 'n03532672', 'n03691459',
                      'n02749479', 'n02105056', 'n02279972', 'n04442312', 'n02107908', 'n02229544', 'n04525305', 'n02102318', 'n15075141', 'n01514668',
                      'n04550184', 'n02115913', 'n02094258', 'n07892512', 'n01984695', 'n01990800', 'n02948072', 'n02112137', 'n02123597', 'n02917067',
                      'n03485407', 'n03759954', 'n02280649', 'n03290653', 'n01775062', 'n03527444', 'n03967562', 'n01744401', 'n02128757', 'n01729322',
                      'n03000247', 'n02950826', 'n03891332', 'n07831146', 'n02536864', 'n03697007', 'n02120079', 'n02951585', 'n03109150', 'n02168699']
        # ord = np.random.permutation(list(range(100)))
        # ord, class_order = zip(*sorted(zip(ord, class_order)))
        dataset = dataset_class(f'{dir}/{data_name}/{type}', transform=dataset_transform, target_transform=target_transform)
        class_to_new_idx = {}
        for it, val in enumerate(class_order):
            class_to_new_idx[val] = it
        old_idx_to_class = {}
        for key, val in dataset.class_to_idx.items():
            old_idx_to_class[val] = key
        target_transform = transforms.Lambda(lambda y, p=class_to_new_idx, m=old_idx_to_class: int(p[m[y]]))
        dataset.target_transform = target_transform
        vals = list(range(100))
        # new_vals = [target_transform(val) for val in vals]
        labs = [old_idx_to_class[k] for k in vals]
        new_vals = [class_to_new_idx[l] for l in labs]
        list1, list2 = zip(*sorted(zip(new_vals, labs)))
        # logging.info(old_idx_to_class[3])
        # logging.info(new_vals)
        # logging.info(labs)
        # logging.info(list1)
        # logging.info(list2)
    else:
        dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                                download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        logging.info(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

#----------------------------------------------------------------------------------------------------------#

def get_singlecontext_datasets(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False, exception=False):
    '''Load, organize and return train- and test-dataset for requested single-context experiment.'''

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[name]
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name+"_denorm"]
    if name!="CIFAR50" and name!='MINI' and name!='TINY' and name!='IN100':
        config['output_units'] = config['classes']
        trainset = get_dataset(name, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
        testset = get_dataset(name, type='test', dir=data_dir, verbose=verbose, normalize=normalize)
    else:
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        # prepare train and test datasets with all classes
        trainset = get_dataset(name, type="train", dir=data_dir, target_transform=target_transform,
                               verbose=verbose, augment=augment, normalize=normalize)
        testset = get_dataset(name, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
                              augment=augment, normalize=normalize)
        classes_per_first_context = 100 if name=='TINY' else 50
        labels_per_dataset_train = list(np.array(range(classes_per_first_context)))
        labels_per_dataset_test = list(np.array(range(classes_per_first_context)))
        trainset = SubDataset(trainset, labels_per_dataset_train)
        testset = SubDataset(testset, labels_per_dataset_test)
        config['output_units'] = 200 if name=='TINY' else 100
    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config
def get_all_data(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False, exception=False):
    config = DATASET_CONFIGS[name]
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name+"_denorm"]
    if name!="CIFAR50" and name!='MINI' and name!='IN100':
        config['output_units'] = config['classes']
        trainset = get_dataset(name, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
        testset = get_dataset(name, type='test', dir=data_dir, verbose=verbose, normalize=normalize)
        classes_per_first_context = 100
        labels_per_dataset_train = list(np.array(range(classes_per_first_context)))
        labels_per_dataset_test = list(np.array(range(classes_per_first_context)))
        trainset = SubDataset(trainset, labels_per_dataset_train)
        testset = SubDataset(testset, labels_per_dataset_test)
        config['output_units'] = 100 
    else:
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        # prepare train and test datasets with all classes
        trainset = get_dataset(name, type="train", dir=data_dir, target_transform=target_transform,
                               verbose=verbose, augment=augment, normalize=normalize)
        testset = get_dataset(name, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
                              augment=augment, normalize=normalize)
        classes_per_first_context = 50
        labels_per_dataset_train = list(np.array(range(classes_per_first_context)))
        labels_per_dataset_test = list(np.array(range(classes_per_first_context)))
        trainset = SubDataset(trainset, labels_per_dataset_train)
        testset = SubDataset(testset, labels_per_dataset_test)
        config['output_units'] = 50  
    return (trainset, testset), config 
#----------------------------------------------------------------------------------------------------------#

def get_context_set(name, scenario, contexts, data_dir="./datasets", only_config=False, verbose=False,
                    exception=False, normalize=False, augment=False, singlehead=False, train_set_per_class=False):
    '''Load, organize and return a context set (both train- and test-data) for the requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first context (permMNIST) or digits
                            are not shuffled before being distributed over the contexts (e.g., splitMNIST, CIFAR100)'''

    ## NOTE: options 'normalize' and 'augment' only implemented for CIFAR-based experiments.
    exception=True
    # Define data-type
    if name == "splitMNIST":
        data_type = 'MNIST'
    elif name == "permMNIST":
        data_type = 'MNIST32'
        if train_set_per_class:
            raise NotImplementedError('Permuted MNIST currently has no support for separate training dataset per class')
    elif name == "CIFAR10":
        data_type = 'CIFAR10'
    elif name == "CIFAR100":
        data_type = 'CIFAR100'
    elif name == "CIFAR50":
        data_type = 'CIFAR50'
    elif name == 'MINI':
        data_type = 'MINI'
    elif name == 'TINY':
        data_type = 'TINY'
    elif name=='IN100':
        data_type = 'IN100'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict
    config = DATASET_CONFIGS[data_type].copy()
    config['normalize'] = normalize if (name=='CIFAR100' or name=='CIFAR50' or name=='MINI' or name=='TINY' or name=='IN100') else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name+"_denorm"]
    # check for number of contexts
    if contexts > config['classes'] and not name=="permMNIST":
        raise ValueError("Experiment '{}' cannot have more than {} contexts!".format(name, config['classes']))
    # -how many classes per context?
    classes_per_context = 10 if name=="permMNIST" else int(np.floor(config['classes'] / contexts))
    if data_type == 'CIFAR50' or data_type == 'MINI' or data_type=='TINY' or data_type=='IN100':
        if data_type=='TINY':
            classes_per_first_context = 100
        else:
            classes_per_first_context = 50
        contexts -= 1
        if contexts > classes_per_first_context:
            raise ValueError("Experiment '{}' cannot have more than {} contexts!".format(name, 50))
        classes_per_context = int(np.floor(classes_per_first_context / contexts))
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes_per_context if (scenario=='domain' or
                                                    (scenario=="task" and singlehead)) else classes_per_context*contexts
    if data_type == 'CIFAR50' or data_type == 'MINI' or data_type=='TINY' or data_type=='IN100':
        config['output_units'] = classes_per_context*contexts + classes_per_first_context
    # -if only config-dict is needed, return it
    if only_config:
        return config

    # Depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # get train and test datasets
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=None, verbose=verbose)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=None, verbose=verbose)
        # generate pixel-permutations
        if exception:
            permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(contexts-1)]
        else:
            permutations = [np.random.permutation(config['size']**2) for _ in range(contexts)]
        # specify transformed datasets per context
        train_datasets = []
        test_datasets = []
        for context_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(
                lambda y, x=context_id: y + x*classes_per_context
            ) if scenario in ('task', 'class') and not (scenario=='task' and singlehead) else None
            train_datasets.append(TransformedDataset(
                trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
    else:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        # perm_class_list = np.random.RandomState(seed=1).permutation(list(range(classes)))
        temp_list = np.argsort(perm_class_list)
        logging.info(temp_list)
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        # prepare train and test datasets with all classes
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
                               verbose=verbose, augment=augment, normalize=normalize)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
                              augment=augment, normalize=normalize)
        # generate labels-per-dataset (if requested, training data is split up per class rather than per context)
        if data_type!="CIFAR50"  and data_type != 'MINI' and data_type!='TINY' and data_type!='IN100':
            labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [
                list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
            ]
            labels_per_dataset_test = [
                list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
            ]
        else:
            labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [set(list(np.array(range(classes_per_first_context))))]+[
                set(list(np.array(range(classes_per_context))+classes_per_context*context_id+classes_per_first_context)) for context_id in range(contexts)
            ]
            labels_per_dataset_test = [set(list(np.array(range(classes_per_first_context))))] + [
                set(list(np.array(range(classes_per_context))+classes_per_context*context_id+classes_per_first_context)) for context_id in range(contexts)
            ]
            
        # split the train and test datasets up into sub-datasets
        train_datasets = []
        for labels in labels_per_dataset_train:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
                    scenario=='domain' or (scenario=='task' and singlehead)
            ) else None
            train_datasets.append(SubDataset(trainset, labels, target_transform=target_transform))
        test_datasets = []
        for labels in labels_per_dataset_test:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
                    scenario=='domain' or (scenario=='task' and singlehead)
            ) else None
            test_datasets.append(SubDataset(testset, labels, target_transform=target_transform))

    # Return tuple of train- and test-dataset, config-dictionary and number of classes per context
    return ((train_datasets, test_datasets), config)