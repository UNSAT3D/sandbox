Exploration of using Aron's library [Groco](https://github.com/APJansen/GroCo) for this project.

It allows for a UNet that is equivariant to rotations and reflections.
If we write the model as M and any rotation of a multiple of 90 degrees and/or a reflection as R, and any input image as I, what this means is:

M(R(I)) = R(M(I))

i.e. we get the same result no matter if we *first* rotate the image and *then* apply the model, or if we *first* apply the model and *then* rotate the result.

The script here shows that the library has all the ingredients necessary to create an equivariant UNet. It builds a simple one and applies it to a test image from MNIST.
The result can be seen in the image below:


