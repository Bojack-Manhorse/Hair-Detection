# Hair Detection Neural Network

We finetune the Mask R-CNN neural network to detect hair from an image of a person's face, and create an API to interact with the model.

Mask R-CNN was developed in this paper: https://arxiv.org/abs/1703.06870, and a PyTorch tutorial (which was used in developing this implementaion) can be found here: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

## Setting up the API:

Before the API can be set up, the model weights must be placed in the `app` folder. Either train the model yourself (see the following section), or download a set of weights here: https://drive.google.com/file/d/1dGojYbe9T5IyDSBmo-gjqKjNpg8Ccj16/view?usp=drive_link.

The file structure within `app` must look this this:

![Image](Readme_Images/File_Structure.png)

Then the API must be run via Docker. Navigate to the `app` folder within a terminal and create a docker image from the dockerfile.

```
cd app
docker build . -t <tage_name>
```

Then run the Docker image:

```
docker run -p 8080:8080 -it <tage_name>
```

To access the API, use the OpenAPI documentation webpage:

```
http://localhost:8080/docs
```

## Training the Model

To be completed.