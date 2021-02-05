import copy
import os

import numpy as np
from skimage import io, transform

import matplotlib.pyplot as plt


def pick_face(FACES_PATH):
    """Randomly pick a blank face to serve as the canvas"""
    n_faces = len(os.listdir(FACES_PATH)) - 1  # dont know if i need this
    return np.random.randint(n_faces)


def get_indices(donor_info, blank_info, feature_blank, feature_donor):
    """ Return indices for altering data

    if feature_blank and feature_donor are the same
        returns an identical mapping
        => e.g., with nose, nose nose gets at the position where it was before

    if feature_blank and feature_donor are not the same:
        switchtes the features
    """
    # TODO: Was passiert hier? Genauer verstehen
    # die indexe für das neue organ werden berechnet durch die originale position + die pixeldaten des donors

    # Random offset to pertubate data
    delta_row = np.random.randint(10)
    delta_col = np.random.randint(10)

    new_organ_indices = [None, None]

    # Extract the column indice
    new_organ_indices[0] = (donor_info['pixels'][feature_donor][0][0]  # Feature pixel data from donor
                            + blank_info['positions'][feature_blank][0]  # Feature position from blank
                            + delta_row)                                 # Random offset

    # Extract the row indice
    new_organ_indices[1] = (donor_info['pixels'][feature_donor][0][1]
                            + blank_info['positions'][feature_blank][1]
                            + delta_col)

    return new_organ_indices

def frankenmerge(frankenstein, organ_indices, donor_info, feature):
    """Merge a new organ into the frankenstein image"""

    frankenstein[tuple(organ_indices)] = (0.2
                                        * frankenstein[tuple(organ_indices)]
                                        + (1.0-0.2)
                                        * donor_info['pixels'][feature][1]
                                        )


    return frankenstein



def generate_exlude(blank_face,
                    blank_info,
                    donor,
                    donor_info,
                    exclude,
                    config,
                    show):
    """ Generate a image that excludes a part"""

    frankenstein = blank_face  # numpy array

    # copy all the features except the "exculde" one from donor to frankenstein
    # from donor_positions to blank_positions
    for feature in config.FEATURES:

        if feature != exclude:

            # Random offset to pertubate data
            delta_row = np.random.randint(10)
            delta_col = np.random.randint(10)


            new_organ_indices = get_indices(donor_info=donor_info,
                                            blank_info=blank_info,
                                            feature_blank=feature,
                                            feature_donor=feature
            )


            # paste to blank
            frankenstein = frankenmerge(frankenstein,
                                        new_organ_indices,
                                        donor_info,
                                        feature)

    if show:
        io.imshow(frankenstein)
        io.show()


    return frankenstein


def generate_switch(blank_face,
                    blank_info,
                    donor,
                    donor_info,
                    switch,
                    config,
                    show):

    frankenstein = blank_face
    feature_1 = switch[0]
    feature_2 = switch[1]

    # fill switch[0] pixels at switch[1] position and vice versa
    new_organ_indices = get_indices(donor_info,
                                    blank_info,
                                    feature_1,
                                    feature_2)

    frankenstein = frankenmerge(frankenstein,
                                new_organ_indices,
                                donor_info,
                                feature_2)

    new_organ_indices = get_indices(donor_info,
                                    blank_info,
                                    feature_2,
                                    feature_1)

    frankenstein = frankenmerge(frankenstein,
                                new_organ_indices,
                                donor_info,
                                feature_1)

    # add the other features normally
    for feature in config.FEATURES:

        # only if its not one of the features that was switched before
        if feature not in switch:
            new_organ_indices = get_indices(donor_info=donor_info,
                                            blank_info=blank_info,
                                            feature_blank=feature,
                                            feature_donor=feature
            )


            # paste to blank
            frankenstein = frankenmerge(frankenstein,
                                        new_organ_indices,
                                        donor_info,
                                        feature)

    if show:
        io.imshow(frankenstein)
        io.show()

    return frankenstein




def init_frankenstein_mask(config, data_chunks):
    # Set up the frankenstein image as a copy of a facless image
    blank_face_index = pick_face(config.NO_FACES_PATH)

    NO_FACE_INDEX = 1
    # First from the chunks select the randomly chosen blank_face_index
    # Then select index 1, which is the blank face
    blank_face = data_chunks[blank_face_index][NO_FACE_INDEX]
    frankenstein_image = copy.deepcopy(blank_face)
    mask = np.zeros(shape=frankenstein_image.shape)

    return frankenstein_image, mask, blank_face_index


def init_donor(config, data_chunks):
    # could and should be merged with frankenstein init
    # Chose the image to copy features from
    organ_donor_index = pick_face(config.ORIGS_PATH)
    ORIGINAL_INDEX = 0
    organ_donor =data_chunks[organ_donor_index][ORIGINAL_INDEX]
    donor = copy.deepcopy(organ_donor)

    return donor, organ_donor_index


def adjust_output(frankenstein_image):
    height = frankenstein_image.shape[0]
    width = frankenstein_image.shape[1]
    bigger = max(height, width)
    smaller = min(height, width)
    adjusted_img: np.ndarray = np.zeros((bigger, bigger, 3))
    offset = int((bigger/2)-(smaller/2))
    adjusted_img[:, offset:offset+smaller, :] = frankenstein_image
    output = transform.resize(adjusted_img, (224, 224))
    return output


def generate_image(config,
                   data_chunks,
                   positions,
                   pixels,
                   predicate,
                   show=False):
    """
    :param canvas: the index of the faceless images to take as background like 0, 1, 2, etc..
    :param organ_mapping: a list of lists like ['nose', 'mouth'] indicating that there should be
        a mouth on the position of the nose
    :return:
    """

    # Get a blank_face as the template for the frankenstein image
    # and a donor to copy the features from
    frankenstein_image, mask, findex = init_frankenstein_mask(config,
                                                              data_chunks)
    donor, donor_index = init_donor(config, data_chunks)

    # original position and pixels for donor and blank_face (frankenstein_image)
    frankenstein_info = {'pixels': pixels[findex],
                         'positions': positions[findex]}

    donor_info = {'pixels': pixels[donor_index],
                 'positions': positions[findex]}

    if show:
        print('[*] Showing empty face')
        print(frankenstein_image)
        plt.imshow(frankenstein_image)
        plt.show()

    if 'has' in predicate:
        """The target predicate is that the image has some part, so now
        we will create it without them"""

        exclude_part = predicate['has']

        image = generate_exlude(blank_face=frankenstein_image,
                                blank_info=frankenstein_info,
                                donor=donor,
                                donor_info=donor_info,
                                exclude=exclude_part,
                                config=config,
                                show=show)


    elif 'top_of' in predicate or 'left_of' in predicate:

        # Extract what to switch
        relation = list(predicate.keys())[0]
        switch = predicate[relation]

        image = generate_switch(blank_face=frankenstein_image,
                                blank_info=frankenstein_info,
                                donor=donor,
                                donor_info=donor_info,
                                switch=switch,
                                config=config,
                                show=show)

    else:
        raise ValueError("Invalid or unknown predicate")



    return adjust_output(image)
