# **Text Summarization**

## 1. Introduction to Text Summarization
Text summarization is the process of generating short, fluent, and most importantly accurate summary of a respectively longer text document. The main idea behind automatic text summarization is to be able to find a short subset of the most essential information from the entire set and present it in a human-readable format. As online textual data grows, automatic text summarization methods have the potential to be very helpful because more useful information can be read in a short time.

There are basically 2 types of text summarization techniques:<br>
- Extractive Text Summarization<br>
  Extractive summarization takes the text, ranks all the sentences according to the understanding and relevance of the text, and presents us with the most important sentences.

  This method does not create new words or phrases, it just takes the already existing words and phrases and presents only that. We can imagine this as taking a page of text and marking the most important sentences using a highlighter.

- Abstractive Text Summarization<br>
  Abstractive summarization, on the other hand, tries to guess the meaning of the whole text and presents the meaning to us. More like a Semantic Analyzer.

  It creates words and phrases, puts them together in a meaningful way, and along with that, adds the most important facts found in the text. This way, abstractive summarization techniques are more complex than extractive summarization techniques and are also computationally more expensive.<br>

Here, I have perform text summarization following the principles of Extractive Text Summarization using the **PageRank function** that is provided in **networkx** package and it is similar as that of TextRank. Ok much of the theory, let's start with the practical implementation.<br><br>
![giphy](https://user-images.githubusercontent.com/67229090/214232539-8e727ee2-dd4f-4e4b-9289-24369ec0b090.gif)
<br><br><br>

## 2. Importing and Analyzing the Dataset
Data is one the most important thing in any Machine Learning or NLP program, becuase **"Data dosen't lie"**. Let's start by having a dataset given below.<br>
[news.csv](https://github.com/ShivamSrng/TextSummarization/files/10487161/news.csv)
<br><br>
Checking if our dataframe has any null values and if they are comparitively lesser, than would drop them. If they are in larger in number them would have to perform capping or imputation. But, the missing values were less, hence considered dropping them:<br>
![image](https://user-images.githubusercontent.com/67229090/214250712-43b47d9e-97e3-4eb3-9d3f-a571db10bf97.png)
<br>
![image](https://user-images.githubusercontent.com/67229090/214250784-ea54ce6e-3c93-407c-975c-7bd7278c86c2.png)
<br><br>
On analyzing randomly selected blocks of data, I realized there are many HTML Tags, HTML entities, punctuations, contractions, and many other undesired features in the dataset. <br><br>
Take a look here, we can see there are many '\n' in the data, which will not contribute to anything, instead it will just deduce the model's performance:<br>
![image](https://user-images.githubusercontent.com/67229090/214233903-ca724222-4e67-408c-9612-dc847b95df02.png)
<br><br>
Also, here's an another block of data, in which we can see tons of HTML tags, entities, extra spaces, tabs, punctuations, etc.:<br>
![image](https://user-images.githubusercontent.com/67229090/214234673-91efc490-a261-4ee0-a8c4-cffb8a18017d.png)
<br><br><br>

## 3. Data Pre-Processing
Data Pre-processing, or simply Data cleaning is the process of reslving the irregularities or undesired data from the the raw data in a way that dosen't reflect the semantic meaning of data. It is most important step in entire project, as data forms the basis for development of vectors. Also, one of the most important thing, **"Order of Data Pre-Processing is also crucial step, so I had to think and act !!!"**<br><br>
Initially I decided to get rid of HTML tags and entities. To do this task, I used **BeautifulSoup** which is a package that is provided by Python, that is majorly used for scrapping data. But here, I am using it to filter the data, as it already provides pre-defined methods for the same. In code implementation, is done using:<br>
![image](https://user-images.githubusercontent.com/67229090/214235911-f34980f0-871e-4e5a-a127-4b89536e101f.png)
<br><br>
Having the HTML tags and entities removed, I decided to filter out '\n' characters and replace them with '' and also decieded to add fullstops to data blocks. Fullstop will play a very important role, at the time of sentence tokenization, which actually splits the string on the basis of fullstop. If a particular datablock has no fullstop, it will be treated entirely as a single sentence. For example, see the data block given below, it has no fullstop:<br>
![image](https://user-images.githubusercontent.com/67229090/214237885-64388979-d7d4-4821-8699-60adbfebed60.png)
<br><br>
So, now had to figure out where to keep fullstops. And this entire thing is handled by a single function as:<br>
![image](https://user-images.githubusercontent.com/67229090/214238073-fac36e9d-567e-4aa2-981b-e65aac9709f0.png)
<br><br>
At this point of time, I found out that the sentences that already had fullstop, will have multiple fullstops after application of above function on them. So to prevent this, it needs to be passed through this function:<br>
![image](https://user-images.githubusercontent.com/67229090/214238491-9f4c6267-34d5-4134-b0eb-a9e1d057f309.png)
<br>
Here **sent_tokenize** function of nltk.tokenize package is the method that will tokenize the entire block of data, based on the fullstop.
<br><br>
I also found out that data had certain contractions like "I'd". So, I decided to process them further and convert them into an actual expanded form, with the help of **contractions** package, which consists of commonly used contractions in natural speech. This task of contraction removal is done by the follwoing function:
![image](https://user-images.githubusercontent.com/67229090/214239124-03912f48-f9d9-49ec-ae8d-ac5862ab64cd.png)
<br><br> 
I decided to make a function that does all Data Pre-processing till here. Remember, I haven't removed the Stopwords yet. This is because, until now the pre-processing we did is general for all the text and it is the data, that I am going to semantically compare with the summary that we are going to get for it at the end. Hence there is no point of removing Stopwords in original data. This task is done using:
![image](https://user-images.githubusercontent.com/67229090/214239680-b1605e9f-62fc-4ad4-b7ac-f2bbe1ad9076.png)
<br><br>
Finally, I reached the step where I planned to remove Stopwords and punctuations. Stopwords removal is done by using the **stopwords** function that is provided in the **nltk.corpus** package. It is done as:
![image](https://user-images.githubusercontent.com/67229090/214240143-7c2fa298-bd68-4469-9e5d-0e831e4a0f47.png)
<br><br>
Combining the entire Data Pre-Processing into a single function as:
![image](https://user-images.githubusercontent.com/67229090/214241736-7259d1a7-1083-46e5-ac4b-67d313d080b3.png)
<br><br>
Although, I was not able to figure out how to remove a link because, I wasn't able to find any regular expression for removing it. I did research on link removal fuctions through BeautifulSoup but wasnt able to do it.
The above function will work perfectly when provided row by row input but at that time we cannot utilize the multiprocessing functions of NLP. Hence creating same function, but this time we shall pass an entire column rather than a row to function. So, for that fucntion is:<br>
![image](https://user-images.githubusercontent.com/67229090/214241991-8fad11d5-cffb-4c7b-8188-de9b08bb0377.png)
<br>
Here **tqdm** is used to show Progress Bar, which gives visual idea how many rows are processed and how many are left.
<br><br>
Now, using spacy I fastened the Data Pre-processing process, by allwoing concurrent processes to execute together in a batch size of 5000.
![image](https://user-images.githubusercontent.com/67229090/214242737-defb44ca-6080-40b7-a74c-f1d2be6cf0f8.png)
![image](https://user-images.githubusercontent.com/67229090/214242793-ecdc7f35-53be-4dfb-bce7-63a652e77a98.png)
Here n_process = -1 indicates that take as many as process to execute together that the processor can handle.
<br><br>
Finally, checking for any errors that might have occured at time of Data Pre-Processing using:
![image](https://user-images.githubusercontent.com/67229090/214243148-95155a3a-db64-40d3-b442-0a54d6580b22.png)
<br><br>
With this, we completed the entire process of Data Pre-Processing:<br>
![giphy (2)](https://user-images.githubusercontent.com/67229090/214243585-db280af4-e5d5-412a-8f1f-0401188b2f09.gif)
<br><br><br>
## 4. Generating the summary for any one random row
For converting words to vector I used Word2Vec function that is provided by gensim.models package.<br><br>
Some theory behind Word2Vec:<br>
In Word2Vec method, unlike One Hot Encoding and TF-IDF methods, unsupervised learning process is performed. Unlabeled data is trained via artificial neural networks to create the Word2Vec model that generates word vectors. Unlike other methods, the vector size is not as much as the number of unique words in the corpus. The size of the vector can be selected according to the corpus size and the type of project. This is particularly beneficial for very large data. For example, if we assume that there are 300 000 unique words in a large corpus, when vector creation is performed with One Hot Encoding, a vector of 300 000 size is created for each word, with the value of only one element of 1 and the others 0. However, by choosing the vector size 300 (it can be more or less depending on the user’s choice) on the Word2Vec side, unnecessary large size vector operations are avoided.<br><br>
Choosing the random data block using:
![image](https://user-images.githubusercontent.com/67229090/214246544-01fbb842-e173-4401-89ca-11d63d9f1639.png)
![image](https://user-images.githubusercontent.com/67229090/214246142-c61457b0-c587-4b0f-86fd-ee7637091c67.png)
<br><br>
To get the tokens, splitting the sentence on the basis of intermediate white space as:
![image](https://user-images.githubusercontent.com/67229090/214246348-86002c67-f861-4fc2-bcc9-42e46c20ef6f.png)
<br><br>
Getting the embedding of words in the sentence as:
![image](https://user-images.githubusercontent.com/67229090/214246732-895ac254-952b-4369-84a6-be62001e3ac4.png)
<br><br>
Taking the words embeddings generated in the previous stage as input for the cosine similarity matrix generation as:
![image](https://user-images.githubusercontent.com/67229090/214246899-c9910b50-39bc-4304-8ca6-ef045ea754bc.png)
<br><br>
Now, based on the similarity matrix generated, a PageRank algorithm classifies which sentence are sematically more important amoung all the sentences. This is coded as:
![image](https://user-images.githubusercontent.com/67229090/214247246-8fe2c2e0-d6f2-40d7-8416-c2bcc4690ed8.png)
<br><br>
If the original content has 12 sentences then our summary will be generated of 4 lines, which is exccatly depicted above.
Finally generateing the summary, by:
![image](https://user-images.githubusercontent.com/67229090/214247585-c7f25404-9de4-4fed-b436-7fc4d7cfe85e.png)
<br><br>
Henceforth, till now, we can generate summary as:
![image](https://user-images.githubusercontent.com/67229090/214247814-efe84cd2-08fa-4ec4-b35b-9bd59b109b1f.png)
<br><br><br>
## 5. Generating summary for entire Dataframe
It is a function that combines each and every step we performed above into a single function, as:
![image](https://user-images.githubusercontent.com/67229090/214248212-45c9c2fa-bc67-43c0-bc8a-551bdce37efa.png)
<br><br>
Let's see if it works when provided any random data block:
![image](https://user-images.githubusercontent.com/67229090/214248492-64aa083b-a98b-4209-9ab3-0e3b8d30c977.png)
<br><br>
![giphy (3)](https://user-images.githubusercontent.com/67229090/214249153-75bcd0e0-d070-403c-bf22-be911770b530.gif)
<br><br>
For faster result generation using spacy library, I framed a column-wise function as:<br>
![image](https://user-images.githubusercontent.com/67229090/214249358-7ef8136e-3da0-466d-88d4-e1ca4b1201a4.png)
<br><br>
It was a long process, took atmost 35 mins of time:<br>
![giphy (4)](https://user-images.githubusercontent.com/67229090/214249694-c4d84a78-7f95-4b47-b56f-efdc69855ea0.gif)
<br><br><br>

## 6. Generating a new Dataframe
It stores the data under Content column of Original Dataframe as Original Content after applying initial pre-processing in it, which removes HTML tags, entities, extra characters, etc.
![image](https://user-images.githubusercontent.com/67229090/214250352-2da2e930-1411-4b23-9437-b914063fcbc8.png)
<br><br>
This is the part, where I faced a major problem, as it was unexpected. For data in some rows (like the one in the below picture: 229th row) the vectors generated were very long. Due to this, I was continously getting error as "Power Iteration failed". At last, I increased maximum interations to 1 Lakh (this is obviously veryyyyy high). But still, error persisted.
<br><br>
Even, I rechecked everything from Data Pre-processing till the point, but I found no more pre-processing of data could be done to reduce the vectors. Eventually, I had to add try-except block in the function, so that if summary for a particular data cannot be generated, just replace it with empty string.
<br><br>
![giphy (5)](https://user-images.githubusercontent.com/67229090/214252021-f1590b34-3908-4533-9023-abbb0e99406e.gif)
<br><br>
Here, after thinking of each and every possibility, I decided to drop the row as its just 1 row, by:
![image](https://user-images.githubusercontent.com/67229090/214251243-62379341-b87d-4596-b503-48f7e265dee1.png)
<br>
![image](https://user-images.githubusercontent.com/67229090/214251316-1d9bb850-e366-4a29-9908-d865d7289d26.png)
<br><br>
Saving, my progress until here as:
![image](https://user-images.githubusercontent.com/67229090/214252288-73d7af70-b45b-4dd6-9ba1-9a035df80467.png)
<br><br><br>
## 7. Getting Metrics ready
We already sorted out the lines which were most important in Semantic Analysis, but I needed to keep track of the lines that were removed, which I waas able to do with follwing function. And at last stored in the datafram, under Removed Line column.
![image](https://user-images.githubusercontent.com/67229090/214252836-773a3f0b-c2a9-4650-a5cd-1e41689b0093.png)
<br>
![image](https://user-images.githubusercontent.com/67229090/214252931-e6aa4c6d-2c9e-4d3e-b87d-78bf3e1d33c9.png)
<br><br>
Now for generating the metrics, I decided to use Cosine Similarity and Semantic Similarity between the Original Content and New Content column's data. For Cosine Similarity, I used the function provided by spacy library which I used earlier.
![image](https://user-images.githubusercontent.com/67229090/214253775-8184d5a3-b37b-4d5e-8a20-0ebece01467b.png)
<br><br>
And for Semantic Similarity, I used the similarity function that is again provided by spacy library.
![image](https://user-images.githubusercontent.com/67229090/214253698-6391e056-072d-400a-a163-46724154393d.png)
<br><br>
Finally, after computation created their seperate columns. And, I realized there was very vast difference in the mean, median values of the semantic similarity generated by both the functions.<br>
![image](https://user-images.githubusercontent.com/67229090/214254103-50dcffea-ebfc-4857-ad6e-2ec1e633b15e.png)
![image](https://user-images.githubusercontent.com/67229090/214254189-8aa801ef-d338-4442-8f7f-de1a95886b56.png)
<br><br>
So, to get a final value, I decided to perform Harmonic Mean of both the values and also add it's respective column in out dataframe as:
![image](https://user-images.githubusercontent.com/67229090/214254623-5c5953d7-8ae7-45de-9fda-941577afe821.png)
<br>
![image](https://user-images.githubusercontent.com/67229090/214254707-7491e8a7-e850-42b6-afea-43b3fe67d429.png)
<br><br><br>
## 8. The Result CSV File
After completeing everything, I made a CSV file of the dataframe on which I was working by:
![image](https://user-images.githubusercontent.com/67229090/214255042-a95321cb-295e-4170-b9a2-0c71432a0b39.png)
<br><br>
Finally, completed the entire task of task summarization, with an average accuracy of **82.66%**.
<br><br><br>
## 9. Result files
+ Input Dataset: [news.csv](https://github.com/ShivamSrng/TextSummarization/files/10488107/news.csv)
+ Jupyter Notebook ipynb file: [Text Summarization.zip](https://github.com/ShivamSrng/TextSummarization/files/10488096/Text.Summarization.zip)
+ PDF of Jupyter Notebook: [Text Summarization - Jupyter Notebook.pdf](https://github.com/ShivamSrng/TextSummarization/files/10488074/Text.Summarization.-.Jupyter.Notebook.pdf)
+ Final Result CSV File: [Final Result.csv](https://github.com/ShivamSrng/TextSummarization/files/10488117/Final.Result.csv)






