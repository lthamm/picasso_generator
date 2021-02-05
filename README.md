# Picasso Theory Test Generator

## Purpose
Adaption of the [Picasso Generator](https://github.com/mc-lovin-mlem/concept-embeddings-and-ilp) for testing the Aleph hypothesis generated for
a Deep Neural Network.

Instead of randomly changing the input images based on the masks, the data
will be created according to the predicates. There is a predefined theory
in predicates.json.

Supported predicates:
* `top_of`, `left_of`: Takes a list with two arguments, that will be swapped
* `has`: Takes a part, that will be removed from the output image

Assume you have a theory `has(mouth)`. Add it to predicates.json as an entry
to the array:
`{"has": "mouth"}`.
Then run the project. If everything goes well, you will end up with images,
that do not have a nose.


## Preparation
Get the FASSEG dataset Frontal 1 / Frontal 2 from [here](http://massimomauro.github.io/FASSEG-repository/).
In the root, create a folder `dataset/fasseg`. Then create 3 subfolders:
* heads_labels: The original mask images from the FASSEG dataset. You need to
colorize the right eye magenta.
* heads_original: The original portrait images
* heads_no_face: The original faces but with the facial features nose, eyes and
mouth removed

Also see [the original repo](https://github.com/mc-lovin-mlem/concept-embeddings-and-ilp).


## Usage
Install the requirements: `pip3 install -r requirements.txt`
Run the project: `python3 run.py`

Arguments:
* `n`: Number of images to generate per class. Default 500.
* `show`: Plot images while creation. Default False.
* `predicates`: Path to the predicates.json. Default in the project root.
