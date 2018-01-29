# **Notes from reviewing and running the code samples **

To run the programs (to see the programs in action), start "jupyter notebook"
from the root drectory where this repository was cloned.
Then clicking on each "step" would open a new window.
For executing each step, use "Shift+Enter."
Some steps can take too long to run: for example, fetching data from
the FindLaw.com website could take > 20 minutes.
Since the HW could work on data already retrieved, skip this step.
Scroll down to see what the code does, instead of running the code.
Once the step is complete, or left incomplete and running in which case
"shutdown" manually, move to the next step and repeat.

1. *Supreme Court Topic Modeling*
    23K cases are input. The text is preprocessed (stopwords removed, lemmatized).
    Then Term Frequency Inverse Documency Frequency (TF-IDF) Vectorization
    is applied to extract unique words (by first identifying the frequency of words in each 
    document and assigning weights - unique words are upweighted).
    Using Non-Negative Matrix Factorization (NMF), the words are clustered into a 
    set of topics.
    Each step is described below in detail.
    * Step 1 - Data Collection & Preparation
       * [1] Scientific packages are imported into Python Kernel
               The packages include Beautiful Soup, requests, regular expressions,
               Pandas, Numpy.
         [2] To create a list of Supreme Court documents and their URLs,
               first the root_url (from FindLaw.com) is provided to the program.
               The website lists cases under each year via (clickable) links.
         [3] This section defines some variables and functions.
               The years variable inits the range of years.
               The Beautiful_soup_grabber fetches data from the website.
               The year_gettter function calls the soup grabber function for each year.
         []  The above function is called and the result is assigned to a variable.
             Besides, assign column names.
         [] Next, check the first 5 rows to check table values and alignment.
         [] Select a specific row.
         [] Pickle the data - Python object serialize to get byte stream
         [] Get number of cases (using numpy shape attribute).
       * Close this window, and in the parent window "shutdown" Step 1
    * Step 2 - Data Collection & Preparation    
      The objective is to get full texts of cases found in the previous step. 
      * [] Scientific packages are imported.
        [] Read in output from previous step.
        [] Split dataframe into three temporaries - the objective is to split requests to caselaw
        [] Collect case details
        [] scrape case details and add to table (and concatenate)
        [] The pickle file (all of the data) is about 600 MB (better not to do this on laptop).
      * Close this window, and in the parent window "shutdown" Step 1
    * Step 3 - Data Processing
       This step uses Scikit, Natual Language Toolkit (NLTK), and SpaCy.
       * [2] importing SpaCy failed with the error "ModuleNotFoundError" and following 
         the procedure at the top leads to further errors, and needs troubleshooting
       This step mainly preprocesses text using NLTK.
       Specifically, state names, case names, common stopwords, people's names
       day of the week, etc. are removed. Non-words, such as case numbers, are removed.
       The text is finally lemmatized, i.e., rootwords are obtained.
     * Step 4 - Topic Modeling Method Testing
       The preprocessed text is modeled using SKlearn, LDA, LSA, and NMF.
       The "textblob" package could not be found.
       The main objective at this step is to find all possible topics by creating a model.
       
       * LDA

       LDA operates on the basis that documents cover only a small set of topics
       and the topics use only a small set of words frequently. 
       
       LDA stands for Latent Dirichlet Allocation. Dirichlet is a distribution over 
       multinomial (distributions). Binomial distribution is the distribution
       of probabilities of getting either of two values. Multinomial distribution is
       the distribution of probabilities of getting one of many values (multinomial).
       Here, the many values are the various topics. As opposed to a Dirichlet distribution,
       which assumes that a document goes with a topic, in LDA, a document could be
       associated with multiple topics.

       The performance of LDA is dependent on the terms in the corpus. To improve
       the performance, the method here employs three techniques: (1) frequency filter
       to remove low frequency words, (2) removing less important words (based on context)
       using a Part of Speech tagger, and (3) running LDA multiple times on batches
       then combining the results.

       Despite all the improvements, LDA turned out to be ineffective for this project,
       because LDA fails to capture relationship between documents.

       * LSA 

       LSA is based on the distributional hypothesis which states that words that are close in 
       meaning will occur in similar text. 

       LSA stands for Latent Semantic Analysis. An occurrence matrix is built that describes
       the occurrence of terms in documents. Then a low-rank approximation is applied to the
       occurrence matrix that essentially merges the dimensions associated with the terms
       that have similar meanings.

       Although this approach is good at drawing together closely related words, it is not as
       useful in topic modeling the cases, i.e., assigning a topic to each document.

       * NMF

        This is the last technique that the code applies for topic modeling. NMF stands for
        Non-negative Matrix Factorization. This technqiue again (like LSA) starts with the 
        tf-idf features (term frequency-inverse document frequency), a document-term matrix
        and a user-specified number of topics, and derives two matrices that are factir matrices 
        of the input matrix. The first matrix is a document x topics matrix, and the second one is
        a topics x terms matrix. 

        It is not clear why NMF yields better results than LSA or LDA.

     * Step 5 - Topic Model Application to Data
        Once a model is obtained in the previous step, run the model against
        entire dataset to label the documents. For e.g., case A is on topic T.  
        The step uses NMF to find dot product of input data and the model. 
        A dictionary of topic components is created and used for indexing data.
        Finally, data for a given number of topics is collected and visualized.

2. *word2vec*
    * The following is a description of the steps in the minimalistic implementation of word2vec
      https://www.github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
      The example implements the Skip-Gram model that predicts source context words
      from target words. The training uses negative sampling or noise-contrastive 
      estimation (NCE) loss, that assigns high probabilities to real words and 
      low probabilities to noise words. This method scales well because computing the 
      loss function depends on the choice of the number of noise words, not the size of
      of the vocabulary.
      https://www.tensorflow.org/tutorials/word2vec gives more details on the approach.
    * Step 1 - Download the data 
      Download data from the provided url if the file is nor present already.
      Read the data into a list of strings
    * Step 2 - Build the dictionary and replace rare words with an unknown token.
      Process raw inputs into a dataset.
      Then capture 4 global states/variables:
      1)  data = list of codes
      2) count = map of words to count of occurrences
      3) dictionary = map of words to their codes
      4) reverse dictionary = maps codes to words
      To reduce memory, delete temporary volcabulatry data structure.
    * Step 3 - Generate a training batch for the skip-gram model.
       Break the input data into batches for further processing taking care of end conditions.
    * Step 4 - Build a skip-gram model.
       Choose a random set of words to sample nearest neighbors.
       Use TensorFlow's graph and other functionality to accomplish the steps.
       Pin computation to the CPU (no GPU implementation in this code).
       Look up embeddings (a value between -1.0 and 1.0) for inputs. 
       Construct the variables for the NCE loss.
       Compute the average NCE loss for the batch.
       Construct the SGD (Stochastic Gradient Descent) optimizer 
       using a learning rate of 1.0.
       Compute cosine similarity between minibatch examples and all embeddings.
     * Step 5 - Train the model
       This step is repeated 100,001 times.
       Run the SGD optimizer built in the previous step on the batch of data.
       Every 10K steps, gather the nearest neighbors and display.
         This step is expensive (indexing the sparse reverse dictionary,
         forming a string, and printing to stdout, can cause performance drop.
    * Step 6 - Visualize / display the distance between the embeddings
      

       
 


