### Class for loading the images in batches
### Xian Lopez Alvarez
### 12/7/2017

import numpy as np
import tools
from params import NDIM_DISC, NDIM_CONT


class data_loader:
    curr_batch = 0
    indexes = []
    nimages = 0
    n_batches_per_epoch = 0
    n_images_per_epoch = 0
    all_classes_in_batch = False
    batch_size = 0
    annotations = []
    shuffle = True
    normalize = 2
    
    def __init__(self, annotations, opts):
        self.shuffle = opts.shuffle
        self.normalize = opts.normalize
        self.annotations = annotations
        self.all_classes_in_batch = opts.all_classes_in_batch
        self.batch_size = opts.batch_size
        self.nimages = len(self.annotations)
        self.n_batches_per_epoch = np.int32(np.ceil(np.float32(self.nimages) / np.float32(self.batch_size)))
        self.n_images_per_epoch = self.batch_size * self.n_batches_per_epoch
        # Build the array with the indexes of all images of one epoch:
        self.prepare_epoch()
    
    
    def prepare_epoch(self):
        # Shuflle indexes
        if self.shuffle:
            self.indexes = np.random.choice(range(self.nimages), self.nimages, replace=False)
        else:
            self.indexes = range(self.nimages)
        # If the number of images is not a multiple of the number of batches, then we add random images to
        # the last batch:
        empty_slots = self.n_images_per_epoch - self.nimages
        if empty_slots > 0:
            if self.shuffle:
                self.indexes = np.concatenate((self.indexes, np.random.choice(range(self.nimages), empty_slots, replace=False)))
            else:
                self.indexes = np.concatenate((self.indexes, range(self.nimages)[0:empty_slots]))
        # If told to provide at least one instance of each class in every batch, do so:
        if self.all_classes_in_batch:
            # We put the shuffled indexes into unused_indexes, and free self.indexes (then we'll fill it again
            # in another order, making sure every batch has at least one instance of each class).
            unused_indexes = self.indexes.tolist()
            n_unused = len(unused_indexes)
            self.indexes = []
            idx_selected = -1
            idx_unused = -1
            # Flags to detect on what batches we have failed finding all the categories:
            batches_failed = np.zeros(self.n_batches_per_epoch, dtype=bool)
            # Flags to detect on what categories we have failed finding them for the batch:
            flags_failed = np.zeros((self.n_batches_per_epoch, 26), dtype=bool)
            # Loop over all batches:
            for batch_id in range(self.n_batches_per_epoch):
                count_in_batch = 0
                # Flags to detect what categories have already been used in the batch:
                used_categories = np.zeros(26, dtype=bool)
                for cat_idx in range(26):
                    attemps = n_unused
                    while not used_categories[cat_idx] and not flags_failed[batch_id, cat_idx]:
                        idx_unused = idx_unused + 1
                        # If we go over the number of unused images, restart from the beginning:
                        if idx_unused >= n_unused:
                            idx_unused = 0
                        # Get the categories of this image:
                        cat_one_hot = tools.get_categories(self.annotations[unused_indexes[idx_unused]])
                        if cat_one_hot[cat_idx]:
                            # If it has the category we are looking for, we add it.
                            count_in_batch = count_in_batch + 1
                            if count_in_batch > self.batch_size:
                                # We should never run into this.
                                tools.error('batch overflow.')
                            idx_selected = idx_selected + 1
                            self.indexes.append(unused_indexes[idx_unused])
                            # Remove from unused list:
                            del unused_indexes[idx_unused]
                            n_unused = n_unused - 1
                            # If the length of unused_indexes is lower now than idx_unused, restart this index:
                            if idx_unused >= n_unused:
                                idx_unused = 0
                            # Check if we ran out of unused images:
                            if n_unused < 0:
                                tools.error('Number of unused images lower than 0.')
                            # Check if other categories are also included:
                            for idx3 in range(26):
                                if cat_one_hot[idx3]:
                                    used_categories[idx3] = True
                        # Check if we have run over all the unused images, and yet failed finding this
                        # category:
                        attemps = attemps - 1
                        if attemps == 0:
                            print('Failed finding images for category %i on batch %i' % (cat_idx, batch_id))
                            flags_failed[batch_id, cat_idx] = True
                # Once here, we have already put an instance of each class in the batch, or failed with this.
                # We fill the rest of entries with random images:
                while count_in_batch < self.batch_size:
                    count_in_batch = count_in_batch + 1
                    if count_in_batch > self.batch_size:
                        # We should never run into this.
                        tools.error('batch overflow.')
                    idx_selected = idx_selected + 1
                    self.indexes.append(unused_indexes[idx_unused])
                    # Remove from unused list:
                    del unused_indexes[idx_unused]
                    n_unused = n_unused - 1
                    # If the length of unused_indexes is lower now than idx_unused, restart this index:
                    if idx_unused >= n_unused:
                        idx_unused = 0
                    # Check if we ran out of unused images:
                    if n_unused < 0:
                        tools.error('Number of unused images lower than 0.')
                # Check if we have failed finding any category:
                if np.any(flags_failed[batch_id]):
                    batches_failed[batch_id] = True
            # Convert to numpy array:
            self.indexes = np.asarray(self.indexes, dtype=np.int32)
            # Print on screen how many batches were successful and how many not:
            print('%i batches have instances of all classes' % np.sum(batches_failed == False))
            print('%i batches do not have instances of all classes' % np.sum(batches_failed))
        
            # Check if the indexes are consistent:
            idx_image = -1
            for batch_id in range(self.n_batches_per_epoch):
                used_categories = np.zeros(26, dtype=bool)
                for _ in range(self.batch_size):
                    idx_image = idx_image + 1
                    cat_one_hot = tools.get_categories(self.annotations[idx_image])
                    for cat_idx in range(26):
                        if cat_one_hot[cat_idx]:
                            used_categories[cat_idx] = True
                for cat_idx in range(26):
                    if not flags_failed[batch_id, cat_idx]:
                        # If we are here, for this batch we found an instance of this category.
                        if not used_categories[cat_idx]:
                            tools.error('Indexes incoherence.')
                    else:
                        # If we are here, for this batch we did not find an instance of this category.
                        if used_categories[cat_idx]:
                            tools.error('Indexes incoherence.')
    
    
    def load_batch(self):
        # Initialize arrays:
        im_full_prep_batch = np.zeros([self.batch_size, 224, 224, 3], dtype=np.float32)
        im_body_prep_batch = np.zeros([self.batch_size, 128, 128, 3], dtype=np.float32)
        
#        print('loading images for batch %i' % self.curr_batch)
        
        # Fill the batch:
        for idx_in_batch in range(self.batch_size):
            # Corresponding image index:
            im_idx = self.curr_batch * self.batch_size + idx_in_batch
            
            if im_idx >= self.n_images_per_epoch:
                tools.error('Image index over number of images per epoch')
            
            # Load images (full and body):
            im_full, im_body = tools.load_images(self.annotations[self.indexes[im_idx]])
            im_full, im_body = tools.load_images(self.annotations[self.indexes[im_idx]], self.normalize)

#            # Convert to numpy arrays:
#            im_full_prep = np.float32(im_full) / 255
#            im_body_prep = np.float32(im_body) / 255
#            
#            # Normalize:
#            for idx_channel in range(3):
#                im_full_prep[:,:,idx_channel] = im_full_prep[:,:,idx_channel] - tools.mean[idx_channel]
##                im_full_prep[:,:,idx_channel] = im_full_prep[:,:,idx_channel] / tools.std[idx_channel]
#                im_body_prep[:,:,idx_channel] = im_body_prep[:,:,idx_channel] - tools.mean[idx_channel]
##                im_body_prep[:,:,idx_channel] = im_body_prep[:,:,idx_channel] / tools.std[idx_channel]

            # Add one dimension (for batch)
            im_full_prep_batch[idx_in_batch, :, :, :] = im_full
            im_body_prep_batch[idx_in_batch, :, :, :] = im_body
        
        # Update batch index:
        self.curr_batch = self.curr_batch + 1
        
        # If we have completed a whole epoch, prepare a new one and restart the batch index:
        if self.curr_batch == self.n_batches_per_epoch:
            self.curr_batch = 0
            self.prepare_epoch()
        
        return im_full_prep_batch, im_body_prep_batch
    
    
    def load_batch_with_labels(self):
        # Initialize arrays:
        im_full_prep_batch = np.zeros([self.batch_size, 224, 224, 3], dtype=np.float32)
        im_body_prep_batch = np.zeros([self.batch_size, 128, 128, 3], dtype=np.float32)
        true_labels_cont = np.zeros((self.batch_size, NDIM_CONT), dtype=np.float32)
        true_labels_disc = np.zeros((self.batch_size, NDIM_DISC), dtype=np.float32)
        
#        print('loading images and labels for batch %i' % self.curr_batch)
        
        # Fill the batches:
        for idx_in_batch in range(self.batch_size):
            # Corresponding image index:
            im_idx = self.curr_batch * self.batch_size + idx_in_batch
            
            if im_idx >= self.n_images_per_epoch:
                tools.error('Image index over number of images per epoch')
            
            # Load images (full and body):
#            im_full, im_body = tools.load_images(self.annotations[self.indexes[im_idx]])
            im_full, im_body = tools.load_images(self.annotations[self.indexes[im_idx]], self.normalize)

#            # Convert to numpy arrays:
#            im_full_prep = np.float32(im_full) / 255
#            im_body_prep = np.float32(im_body) / 255
#            
#            # Normalize:
#            for idx_channel in range(3):
#                im_full_prep[:,:,idx_channel] = im_full_prep[:,:,idx_channel] - tools.mean[idx_channel]
##                im_full_prep[:,:,idx_channel] = im_full_prep[:,:,idx_channel] / tools.std[idx_channel]
#                im_body_prep[:,:,idx_channel] = im_body_prep[:,:,idx_channel] - tools.mean[idx_channel]
##                im_body_prep[:,:,idx_channel] = im_body_prep[:,:,idx_channel] / tools.std[idx_channel]

            # Add one dimension (for batch)
            im_full_prep_batch[idx_in_batch, :, :, :] = im_full
            im_body_prep_batch[idx_in_batch, :, :, :] = im_body

            # Build the batch with the true labels:            
            # Discrete:
            for cat_idx in range(NDIM_DISC):
                if tools.category_in_annotation(self.annotations[self.indexes[im_idx]], cat_idx):
                    true_labels_disc[idx_in_batch, cat_idx] = 1
            # Continuous:
            for var_idx in range(NDIM_CONT):
                true_labels_cont[idx_in_batch, var_idx] = np.float32(self.annotations[self.indexes[im_idx]][6 + var_idx]) / 10
        
        # Update batch index:
        self.curr_batch = self.curr_batch + 1
        
        # If we have completed a whole epoch, prepare a new one and restart the batch index:
        if self.curr_batch == self.n_batches_per_epoch:
            self.curr_batch = 0
            self.prepare_epoch()
            
        # TODO: divide continuous true labels by 10?
        
        return im_full_prep_batch, im_body_prep_batch, true_labels_cont, true_labels_disc










