# hello_signal
**Rich Pang**

2026-05-23

A hello_signal file is a single file, usually a PDF, that contains a minimal description of a dataset and a minimal demonstration of how to access the data and clearly display one or more signals thought to be present in it.

## hello_signal files:

Make it easy to quickly estimate the complexity of a dataset. Does the data use standard formats, or are bespoke file types and software involved? Will this data require a GPU cluster or can I analyze it on my laptop? 

Expedite on-boarding newcomers to specific data, such as new lab members or computational collaborators, by eliminating time and errors in the data munging process. It's very easy to make small errors when loading and pre-processing someone else's data, even with detailed instructions on how to work with it. A hello_signal file demonstrates precisely how to load the data and confirm it looks how it's supposed to to minimize errors.

Are a great way to organize a lab's datasets, which may be scattered across hard drives, computers, and the cloud, along with inconsistent or partial documentation. Making hello_signal files for these dataset doesn't require any reorganization of existing data. A directory of hello_signal files for a lab's datasets makes an excellent gateway to the data and can dramatically increase its effective half-life.

## Structure of a hello_signal file

#### Filename

A useful format:

YYYYMMDD_hello_datasetname_signalplotted.pdf

#### 1. Brief description of the nature, structure, and dimensions of the data

Roughly what is in the data? Spike trains? Tracked animal behavior? BOLD signals?

What is the associated publication, if any?

How is the data structured? How many files are there? How big are they?

#### 2. Instructions for downloading or accessing the data

How does one download or access the data?

#### 3. Software used

Software/packages + versions used to run the code below.

#### 4. Minimal code for plotting clear signal(s) in the data

Minimal code for opening data and plotting signal(s) of interest.
This includes extracting relevant signals, aligning timestamps, etc.
This should be plain text that one can simply paste into a code editor and run.
The clearer and more convincing the signal displayed the better.
For instance, plotting a single spike train is worse than plotting several spike trains clearly responding to a stimulus presented in the dataset.

#### 5. Outputs

Visualizations expected from running the above.
This is to confirm that the code was run correctly.

#### 6. Notes

Sections of the data that may be corrupted, artifacts to be aware of, etc.
Other elements of experimental context that are important to know about.

## Example hello_signal files

* [20260525_hello_flywire2024_single_neuron_swc_morphology.pdf](20260525_hello_flywire2024_single_neuron_swc_morphology.pdf)

## Who should create hello_signal files

A hello_signal file can often be put together from a dataset given sufficient documentation and familiarity of the person compiling the hello_signal file with the type of data involved.
The person who can generally do this the fastest is the person who collected the data.

## Other benefits of hello_signal files

* No specialized software involved. Very flexible.
* Quite easy to make a hello_signal file if one is familiar with the data.
* Works with both in-house and public datasets.
* Lightweight yet robust. Can even be printed out and put on the fridge.
* Only break if the dataset is moved. For this reason it is useful to archive the data in a specific place.
* Works in concert with highly heterogeneous datasets instead of fighting to standardize them.
* Can likely be produced with LLMs given a collection of partial documentation (but should be verified).
* Greatly reduce overall time spent data munging, creating more time for science. 