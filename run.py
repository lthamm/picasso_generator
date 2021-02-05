import os
import json
import argparse

from skimage import io, img_as_ubyte
from tqdm import tqdm

from image.generate import generate_image
from image import chunks

parser = argparse.ArgumentParser(description='Create picasso dataset images to test an aleph theory')
parser.add_argument('--n', default=500, type=int,
                    help=('Number of images to create for every predicate'))
parser.add_argument('--show', default=False, type=bool,
                    help=('Plot images while generation.'))

parser.add_argument('--predicates', default='./predicates.json',
                    help=('Json file that contains the predicates to test'))


if __name__ == "__main__":

    import config
    args = parser.parse_args()

    n_samples = args.n
    show = args.show

    data_chunks, pixels, positions = chunks.make(config)

    with open(args.predicates) as predicates_file:
        predicates = json.load(predicates_file)

        # for each predicate, generate n_samples
        for predicate in predicates:

            # Make a name for the predicate based on the name + arguments
            predicate_name = list(predicate.keys())[0]
            predicate_name += "_" + str(predicate[predicate_name])

            pred_path = os.path.join(config.PICASSO_ROOT,
                                   'not_'+predicate_name, 'neg')

            print(pred_path)
            # Create the output folder
            os.makedirs(pred_path, exist_ok=True)

            for j in tqdm(range(n_samples)):

                try:
                    generated_image = generate_image(config,
                                                     data_chunks,
                                                     positions,
                                                     pixels,
                                                     predicate,
                                                     show=show)

                    generated_image = generated_image.astype(float)
                    generated_image /= 255.0

                    im_path = os.path.join(pred_path,
                                            "pic_" + str(j).zfill(5) + ".png")

                    io.imsave(im_path, img_as_ubyte(generated_image))

                except Exception as e:
                    print(e)
