# Original Lecture Notes: How Machines See
## Voice Memo Transcription and Brainstorm

**Date**: November 2025
**Instructor**: Prof. Joseph Bakarji
**Course**: Introduction to Data Science and Computing

---

## Opening Thoughts

Good morning everyone. So today we'll talk about **how machines see**. The question is, what is seeing for a machine? And believe it or not, we understand machines seeing more than we understand how we see things. Why? **Because we built them, obviously.**

Now there's a long history of studying vision, not only from a machine perspective, but also from a human perspective. What is human vision? You speak of retina, of light, vision involves light, it involves reflections, brightness, contrast, colors, and the like. All of these concepts are things that we're familiar with before we invented machines. And when we did invent machines, as humans I would say, we adopted, of course, a lot of concepts that we started first from intuition, from let's say older philosophies and sciences, and then maybe to modern science, physics, about optics, and then we got to the machines.

So it's good to have this sort of background in mind. And to a lot of you, you're already familiar with that intuition, like for example color. If we see color, we'd want machines to see color as well. But the way machines see color, there's a lot of similarity between how they see color and how we do. And there are a lot of differences as well. In both cases, it's some sort of electromagnetic wave that hits your retina or hits the retina of a lens of a camera, and then gets transformed into something that we can cluster, for example, as a color, like blue, red, yellow, and etc.

---

## Lecture Structure

### Part 1: Understanding Digital Images

So what I want to talk about today is how machines see. Vision. We're going to start from a very specific example, or a very specific data set, just like we always do, on what machines do when they see. And then we look at more general examples of data sets on how people see.

#### Student Handwritten Digits

So one thing you've done before this class to prepare for it is that you've **submitted handwritten pictures**, and our TAs have segmented them and have turned them into these images that you see here on the screen. So what happened is you're given a paper with squares, and you wrote a number in that square. And those numbers, also you were asked to write letters in those squares. And those numbers and letters represent different things, right?

So if I show you a number, you can tell me what is this? Number 9? What is this? Number 5? So they're single digits and also the same thing for the letters.

Now what we did is we took a picture of all of these, we scanned them, and we **segmented them** into wherever you have the bounding boxes. So there was some processing that happened. And this processing is part of what you should also learn how to do. But for now, this is what's happening in the background. Actually, we have the code for it. You can go and check what the TAs did in the repository on the website.

#### What Are Pixels?

And once we've segmented all of these pictures, then we ask ourselves the question, first of all, **what are these things?** What are we segmenting? This is the first thing you need to understand for vision.

So the easiest thing to understand, this is a **grid**. If you zoom in, let's zoom in, zoom in a little bit, and then zoom in a bit more, and then zoom in a bit more. **These are pixels**. You've heard of pixels.

When I was your age or younger, teenager and maybe in my 20s, people obsessed about how many megapixels a camera has. Nowadays, we don't talk about number of pixels in an image. These things are much more powerful. Now you have multiple cameras that are taking pictures on your phone. But imagine at the time, let's say this Nokia had how many pixels? Who can guess?

I had to look it up. So this is the number of pixels that Nokia has. Nokia, maybe you haven't heard of Nokia. So I'm that old now, unfortunately.

So you can detect the pixels, you can detect the contrast. And okay, if we zoom in, we see these pixels. In black and white, it's pretty straightforward. **It's some number that is between 0 and 255.**

You can tell me why 255? **Why is 255 so special?** Because it's 2 to the power of 8. It's what's called the **byte**. And a byte can encode numbers that are from 0 to 255. A byte is 0s, 0, 1, 2, 3, 4, etc. to 255. We'll talk about that later, but keep that in mind. But it's basically this number. And then the closer it is to 255, the more it is white. The closer it is to 0, the more it's black. So this is basically what it is.

#### Color Images: RGB

So we have this encoding, we have these pixels, and now we can look at specific numbers. Let's look at one of the numbers that was written by one of you. So let's look at this number 3. So number 3, this looks like 3. We can zoom in, we can look at the squiggles, etc.

And what we want to do is we want to **train a model**. We want a machine to learn how to identify these numbers. So the first thing to think about is how could a machine process numbers?

Now, you know, so when it comes to colors, the encoding is usually, you use **three colors, red, green, and blue**. And now instead of having one grid, you have three grids, and each grid tells you how much red there is, how much green there is, and how much blue. And just like when you paint, you mix colors, and you come up with a new color based on the mixture. Here we're mixing light sources, and then based on the mixture, you come up with a solution, right? All TVs used to work that way. If you looked closer, it was red, blue, green grids. So that's that.

---

### Part 2: Geometric Approaches to Recognition

And all right, so let's see, can we identify these things? All right, so one algorithm would be to have sort of a narrow band.

#### Brainstorming Detection Strategies

You know, you can come up with ideas. **Let's come up with ideas on how to identify specific numbers.** Let's say this specific algorithm will tell me that whether it's more likely to be one of the 10 numbers, right?

**So I gave you the idea about the zeros.** You start from the center, and then you go out. What about other ideas? For example, a band. You know, I have this idea that that's like two parallel lines. They're separated by, you know, a few pixels, let's say 5 to 10 pixels or something like that. And then you sweep through the whole image, and you see if at some point you get a spike in that, right?

Would you code that, actually? You have a band, you sweep through the whole image, and you sort of get a spike in the color. So one thing you can do is you can **sum all of the columns**, right? You sum, you have 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, something like that. And then you get a spike, and maybe we need an algorithm to detect the spike.

**How do you detect the spike?** Maybe it crosses a certain threshold. If it crosses a certain threshold, then it is the number one. And, you know, maybe number 7 also has that spike. Maybe number, I don't know, 8 could have more of a band.

So let's look at the different numbers and how they give us that spike. So we can sweep through them, and look, you can see that number 1 tends to have more of that spike than the others. Actually, it depends how you want to look at that spike and how you want to quantify it.

So that could be one way, and then maybe you have a spike detector, and based on the spike, you now know it's either 1 or not a 1. And this would be part of your algorithm, but it's just a **hard-coded algorithm**. You have to fine-tune certain things.

#### The Experimental Nature

And again, **this is the first lesson in that it's very experimental**. You try things, you come up with ideas, you test them out, and then you step back and you see how well it is, how well it does. And, you know, one of the things, you know, the homework that you have is you're going to have to actually try to see that spike and come up with other ideas on how to detect these numbers.

Now, okay, we also have letters, et cetera, et cetera. Okay, so we go over all, you know, in many computer vision algorithms.

#### Traditional Computer Vision Techniques

I think this is a side note for myself. We go over all these computer vision algorithms. We mention a few, and the purpose is to give students a sense of what it means to work with images, what it means to work with images.

And that could be, and then maybe we start with the letters, and then we move on to the, you know, to the other types of data that are vision, that, you know, we start with letters, and then we look at actual prints of Arabic writings that we want to transcribe. We look at **transcription of notes**. We look at the, again, you know, **videos, which are sequences of images** and how people have to navigate through videos and sequences of images.

And then we look at the data itself in each one of the cases. We, you know, look at the types of processing that people would be interested in, which is, you know:
- Edge detection
- Segmentation
- Processing
- Storage
- Compression
- Downsampling into less pixels
- Upsampling into more pixels

Right? And there are a lot of these. We can demonstrate them on specific examples and show always on the side what the code looks like, right? These are all **notebooks** that would say, okay, here's an image, here's how you process it, here's an image, here's how you process it. And they could come with a video on, you know, what it means to process an image and work with it.

So that would be the **first part and maybe the first lecture** if we're talking about two lectures in the week to deal with how computers see.

---

### Part 3: Modern Machine Learning Approaches

The second lecture would involve working more specifically with **machine learning algorithms that see**, because modern seeing is really that type of seeing.

And the idea would be to, you know, given a machine learning algorithm or some algorithm for detection of letters or for detection of colors, to sort of **use it and see what it gives you as a black box** without really understanding what it's doing. But get a sense of what it means for a machine to process images.

#### MNIST and ImageNet

And, you know, we also go back to the original problem of the MNIST data and the students' data, which is to look at the specific MNIST data and then we ask, you know, and I already have a notebook on the convolutional network example.

And now you can ask questions that are a little more sophisticated in the sense that it's not only about detecting edges, about detecting, you know, segmenting what's in the image, but we can speak of **transformations** that you can do to the image that will give you a sense of, you know, the relationship between the images themselves.

#### Dimensionality Reduction

For example, one thing you can do, you can do **dimensionality reduction**. You can take this whole image, you can take it to a new space that is two-dimensional, right? Instead of having 28 by 28 pixels, I have two numbers that represent these 28 by 28 pixels and that are supposed to represent the differences and the similarities between the images themselves.

So we can speak of **PCA**, which is a linear dimensionality reduction. And we can speak of **SVD**, etc. So these are transformations.

And then we can say, you know, we can speak of **non-linear dimensionality reduction**, like an **autoencoder**, where we go to a non-linear space where things are similar or not.

#### Explaining to Beginners

And how do you explain that to someone who has never actually encountered these concepts? Well, the idea is to start from something simple and to say, well, let's think about this idea, for example.

So if we have a bunch of like this line, which have points, you know, next to each other, what you can do is you can say that I can transform this line into a new type of line by looking at its nearest neighbors and seeing that I can, you know, like this is actually a string, sort of a **rubber band** that I can move around. If I move it around in that way, it becomes a line instead of being a squiggle. So that would be, you know, at least one way how dimensionality reduction works.

Another way how dimensionality reduction works is... So that would be, or in a way it's a transformation. It's a **way that you can take you, it's a function** that can take you from one space to another.

#### The Power of Representation

And so these transformations allow you to say, well, this thing, if I just get a vector out of it and I only work with this grid, for example, of the image, you know, I'm not going to be able to tell you what's one or what's zero by just looking at the grid. But what I can do is I can **go to a new space where it becomes way easier for me to tell you** what is one or zero or, you know, whatever is in that space.

So it becomes a matter of recognizing... So now the question becomes, **how can we go into a new space where things are easier and simpler?**

And again, one way to go into a new space is to go to a much smaller **lower dimensional space**. So these are 100 dimensions. This is called dimensions or coordinates. You can go to a much smaller dimensional space where things make more sense. And these are our points. So now we're talking a little bit more abstractly. But this idea of going to a lower dimensional space is extremely useful. And I'll show you how.

So we go from here to here, and it's some transformation you have to learn about later in a later course. And so this sort of transformation... So let's look at the different types. There are different ways to do that.

#### Drawing Boundaries

And now at this point, we're not identifying, you think, anything. But we're doing the transformation first. And once we do the transformation, we can **draw boundaries** in this new space where we can see, oh, look, there's a boundary between the ones and the zeros in this new space, and I can identify them.

Now you can say, you know, this point was originally this big image, one before. So in this new space, if I want to identify whether it's a one or a zero or a two, all I need to do is to go to this new space and draw a boundary, and I have my number. Right?

And for these transformations, there are a lot of really good transformations that do that for you. And they're a little, you know, mathematically involved, and you'll see them hopefully later in your education. But sometimes, and at this point, **all you need to know is that they're available to you**. And here's an example on how you'd call it in Python.

So I would take an image, I would call this function that projects me into a lower-dimensional space, and now I'm in this new space where things are clearer, and I can just draw the boundary where I need to draw the boundary.

There's this other transformation that seems to be nicer. It seems like it has better luck in actually drawing boundaries. Let's look at this other transformation. Let's look at this other transformation. And then you have these boundaries.

#### Failure Analysis

Now a **very important process** of this exercise, which is to look at where, whatever approach you adopt, whether it's the spiking approach we looked at before, or the sort of kind of transformation approach that we're looking at now, it's very important to **understand where this is going to fail**.

And you do that, well, first thing you can quantify is how many numbers it identifies correctly, you know, out of all the examples, that's the **accuracy**. You can look at also things that it identified as five, even though they were one, and you can plot that in a table, and this is called the **confusion matrix**. How many it confused with other numbers.

But also you can look at **specific examples**. It's very useful to look at specific examples and to see why it didn't get that right. So we can look at that example and say, oh, wait, this seems like it looks a lot like a five. That's why it identified a five. It's a six, but it looks a lot like a five, or it's a one, but it looks a lot like a seven. And you can see how it could be confusing.

And so sometimes you're not going to get 100% accuracy, not because the algorithm is not very good, but just simply **because the data is bad**, that a human, even a human, you know, wouldn't be able to figure out what that handwriting is. And this is when, you know, you write an exam, and I'm there correcting, and I have no idea what you wrote, and I give you a zero. Maybe a machine would give you a better grade, but that's why you have to have a good handwriting.

#### Modern Applications

So again, let's look at specific examples where this is extremely effective. So now **modern machine learning or AI algorithms**, what they do is they take these, they're really good at seeing things, and so they can identify an image, segment an image, and you can identify now, for example, you can say, ask questions like, can I learn what each object is in an image and label it? Like let's say this is a pen, this is a kettle, this is etc., etc.

And in fact, the whole **modern AI revolution** led to all of this boom in the industry and these chat GPT and stuff is because of **AlexNet and ImageNet**. These are different networks or functions. Basically, they're functions. **Anytime we're going to talk about neural networks, I'm just going to call them functions** because this is essentially what they are. They take an input X and they return output Y. They're functions that take an image in and return the label. This is a pen, this is a kettle or whatever it is, right? So that's all you need to understand to really work with these functions. You can treat them as **black box functions** as well. Right? So that's the first step.

#### From Classification to Description

So the second step is you're making these transformations and you're essentially mapping, finding a function that will allow you or allow a machine to actually not only go from an image to tell you whether it's number one or two or three, but **describe the image**. So this is what machine learning algorithms these days do.

So the **input is an image**, but the **output now is a whole sentence**. It's a sentence that describes what's in the image. And then you can see these types of algorithms that sort of find relationships between these images.

Now, an image itself, when we're speaking of relationships in an image, we are speaking of, again, you're localizing elements in an image and you're looking at them.

---

## Homework and Projects

So I think, again, once this lecture, and this would be the second lecture, concludes, then we have ourselves a **homework for students working with images**.

And maybe it will involve them **taking pictures and sharing them** and then creating their own data set and getting a sense of what's involved in processing an image, compressing it and things like that. So it's sort of this **mini project** that they work on and there will be a lot of TAs that assist them with it.

So, yeah, that's it.

---

## Teaching Philosophy Notes

### Key Pedagogical Points

1. **Start with what's given (real data)**
   - Student handwritten digits
   - Real messy, imperfect data

2. **Throw students into the mess**
   - Don't explain everything upfront
   - Let them encounter confusion
   - Build understanding through experimentation

3. **Experimental approach**
   - Try things
   - See what works
   - Analyze failures
   - Learn from mistakes

4. **Progressive complexity**
   - Pixels → Geometric detection → Features → Transformations → Modern ML

5. **Multiple approaches**
   - Hand-coded rules (spike detector)
   - Classical ML (features + SVM)
   - Modern DL (CNNs, autoencoders)

6. **Failure is pedagogical**
   - Confusion matrices
   - Ambiguous cases
   - "Even humans can't solve these"
   - Epistemic humility

7. **Black boxes are okay (at first)**
   - "Here's a function, it does X"
   - Understanding comes later
   - Focus on concepts, not implementation details

8. **Visual and interactive**
   - Zoom into pixels
   - See transformations
   - Plot results
   - Interactive notebooks

---

## Technical Topics Covered

### Lecture 1: Traditional CV
- Digital image representation
- Pixels and encoding (grayscale, RGB)
- Geometric pattern detection
- Edge detection
- Feature extraction
- Classical ML classification
- Confusion matrices
- Failure analysis

### Lecture 2: Modern ML
- Dimensionality reduction (PCA, SVD, t-SNE)
- Autoencoders
- Latent space representations
- Transformation functions
- Decision boundaries in reduced spaces
- Neural networks as functions
- ImageNet and modern breakthroughs
- Image captioning

---

## Notes for Future Development

- Add Arabic text recognition examples (culturally relevant)
- Include video analysis (sequences of images)
- Discuss compression and storage
- Cover upsampling/downsampling
- Connect to modern applications (object detection, segmentation)
- Mini-project: Students create their own datasets
- TA support for data collection and segmentation

---

**End of Original Notes**
