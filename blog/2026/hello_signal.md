# hello_signal
**Rich Pang**

2026-05-25

A hello_signal file is a single file, usually a PDF, that contains a minimal description of a dataset and demonstration of how to access the data and clearly display one or more signals thought to exist in it.

## hello_signal files:

Are fast and easy to make, especially if you're already working with the data.

Enable one to quickly estimate the complexity of a dataset. Does the data use standard formats? Are bespoke file types and software involved? Will this data require a laptop or a GPU cluster to analyze? 

Expedite on-boarding newcomers to data, such as new lab members or collaborators, by eliminating time and errors in the data munging process. It's very easy to make errors when loading and pre-processing someone else's data, even with detailed instructions. A hello_signal file clearly demonstrates how to load the data and confirm it looks how it's supposed to.

Are a great way to organize a lab's datasets, which may be scattered across hard drives, computers, and the cloud, along with inconsistent or partial documentation. Making hello_signal files doesn't require reorganizing existing data, but a directory of hello_signal files can make an excellent gateway to a lab's data and greatly increase its half-life.

## Structure of a hello_signal file

#### Filename

A useful format:

YYYYMMDD_hello_datasetname_signalplotted.pdf

#### 1. Brief description of the nature, structure, and dimensions of the data

Roughly what is in the data? Spike trains? Tracked animal behavior? BOLD signals?
What is the associated publication, if any?
How is the data structured? How many files are there? How big are they?

#### 2. Instructions for downloading or accessing the data

Precise, unambiguous instructions for retrieving the dataset needed to run the code that follows.

#### 3. Software used

Software/packages + versions used to run the code below.

#### 4. Code

Minimal code for opening data and plotting signal(s) of interest.
This includes extracting relevant signals, aligning timestamps, etc.
This should be plain text that one can simply paste into a code editor and run.
The clearer and more convincing the signal displayed the better,
for instance several neurons' spike trains clearly responding to a stimulus.

#### 5. Outputs

Visualizations expected from running the above to confirm the everything is correct.

#### 6. Notes

Sections of the data that may be corrupted, artifacts to be aware of, etc.
Other elements of experimental context that are important to know about.

## Example hello_signal files

* [20260525_hello_flywire2024_single_neuron_swc_morphology.pdf](20260525_hello_flywire2024_single_neuron_swc_morphology.pdf)

## Who should create hello_signal files

A hello_signal file can often be put together from a dataset given sufficient documentation and familiarity of the person compiling the hello_signal file with the type of data involved.
In general the person who can do this the fastest is the person who collected the data.

## Other benefits of hello_signal files

* No specialized software involved. Very flexible.
* Works with both in-house and public datasets.
* Lightweight yet robust format. You can even print it out.
* Only break if the dataset is moved or required software becomes inaccessible.
* Works in concert with heterogeneous datasets instead of fighting to standardize them.
* Greatly reduce overall time spent data munging, creating more time for science. 
