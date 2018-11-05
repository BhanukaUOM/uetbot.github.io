CÁC VẤN ĐỀ HIỆN ĐẠI CỦA CNTT

# [I. Natural Language Processing](https://www.coursera.org/learn/language-processing/home/welcome)

1. [Week 1](https://www.coursera.org/learn/language-processing/home/week/1)

1. Introduction to NLP and our course

* [About this course](https://www.coursera.org/learn/language-processing/lecture/akWLW/about-this-course)

<table>
  <tr>
    <td>Hey everyone. My name is Anna. And I'm here to present our new online course about natural language processing. NLP tasks are everywhere around us; suggest in search, automatic Gmail replies, machine translation. But one task which is especially popular today is chatbots or dialogue systems. It could be a bot that tries to hold the human-like conversation with you. Or it could be a bot that assists with some particular tasks, like comparing two vacation packages, or setting alarms, or answering your questions about credit in the bank instead of a representative in a call center. Do you know how this things work? So what is inside? Some deep learning magic or a rule-based approach or something else. Well, you will build your own chatbot in our course project. But before that, we will cover also some other interesting tasks like text classification, name entity recognition, duplicates detection and many more. Our course is rather advanced. We expect you're already familiar with some methods of machine learning and deep learning, and now you want to apply them to texts. Our goal is not to stack several black box techniques to build a dialogue monster. On the opposite, we will see what is inside each box and cover it with a good amount of mathematics. So does that thing look familiar to you? Are you scared with this letter? Well, you're about to see a lot of this kind of formulas in the upcoming weeks. So, be prepared. We will discuss how to represent pieces of text with sound vectors so that we can compute similarity between vectors. For example, how do you teach a machine that leaky faucet and tap is dripping are kind of the same thing. Personally, I am passionate about NLP. I'm finishing PhD. on that, teaching NLP at Yandex School of Data Analysis and Moscow State University and doing internships and companies from time to time. I have never done online courses though. So, I am very excited to have this experience now, and I will be happy to see you on board.</td>
    <td>Khóa học sẽ dạy về xử lý ngôn ngữ tự nhiên. Trong khóa học sẽ có thực hành tạo chatbot, phân loại văn bản, nhận dạng tên thực thể, xác định lặp, ... 
Người học nên có một chút kiến thức về học máy và học sâu thì sẽ dễ dàng hơn.</td>
  </tr>
</table>


* [Prerequisites check-list](https://www.coursera.org/learn/language-processing/supplement/7p5Lc/prerequisites-check-list)

<table>
  <tr>
    <td>Welcome to our "Natural Language Processing" course! We are excited to have you in the class and we are looking forward to your contributions to the learning community.
In this course you will learn how to solve common NLP problems using classical and deep learning approaches.
Please note that this is an advanced course and we assume basic knowledge of linear algebra, probability theory, machine learning, and deep learning (consider completing "Introduction to Deep Learning" course https://www.coursera.org/learn/intro-to-deep-learning in our specialization). The following incomplete list of terms can help you to get an idea of the theoretical background we expect.
Quick prerequisites check:
Product rule, sum rule, Bayes's theorem, likelihood maximization
Classification, clustering, and regression tasks in machine learning
Loss functions, training vs inference, overfitting problem
Optimization techniques, e.g. (stochastic) gradient descent
Deep Learning architectures, e.g. Recurrent and Convolutional Neural Networks
Python programming and willingness to learn new tool, e.g. Tensorflow.
From a practical side, we expect your familiarity with Python, since we will use it for all assignments in the course. Two of the assignments will also involve TensorFlow. It's OK if you do not have much prior experience with it, but in this case you might expect to spend some extra time on familiarizing yourself with the provided tutorials. Generally, it can take longer than expected to accomplish some tasks - please, take it easy. It means you learn new (and advanced) things, which is great.
Watch the first module to get an overview of what you will learn throughout the course. You might come across some new terms there, but don't worry - we will have 5 more weeks to break them down!
Good luck as you get started and we hope you enjoy the course!</td>
    <td>Trong khóa học này sẽ giải quyết những vấn đề cơ bản của NLP sử dụng các cách tiếp cận cổ điển và học sâu.

Điều kiện tiên quyết:

- Quy tắc cộng, quy tắc nhân và nguyên lý Bayes, hợp lý cực đại
- Phân lớp, phân cụm và hồi quy trong học máy
- Hàm mất mát, huấn luyện với suy luận, vấn đề quá khớp
- Các kỹ thuật tối ưu, ví dụ như xuống dốc (ngẫu nhiên)
- Kiến trúc học sâu, RNN và CNN
- Lập trình với Python và mong muốn học công cụ mới (Tensorflow)</td>
  </tr>
</table>


* [Hardware for the course](https://www.coursera.org/learn/language-processing/supplement/ZpkGK/hardware-for-the-course)

<table>
  <tr>
    <td>In this course you will sometimes train neural networks from scratch and it will take hours, so you need to decide what hardware you want to use.
All assignments can be done on CPU thanks to the architecture choices, but it still takes several hours to train. Using a GPU can reduce training to minutes.
We recommend you the following options:
Google Colab (cloud). It's Jupyter Notebooks from Google. The great thing about them is that they provide free GPUs as of August 2018. It has the latest version of TensorFlow and some things may change in future TensorFlow versions, so you need to be prepared to file an issue on our GitHub or on forums so that we can upgrade our Notebooks. As of August 2018 all our Notebooks are compatible with Google Colab. It's pretty easy to start working on Colab, just follow our instructions: https://github.com/hse-aml/natural-language-processing#running-on-google-colab
Your own hardware. If you have a powerful computer at home you may end up with your own offline setup, please follow the instructions here: https://github.com/hse-aml/natural-language-processing/blob/master/Docker-tutorial.md
Regardless of the choice you make, please make backups of your work!</td>
    <td>Trong khóa học này, bạn sẽ thỉnh thoảng huấn luyện mạng nơ ron từ đầu và nó sẽ tốn rất nhiều thời gian, vì vậy bạn nên quyết định phần cứng mà bạn muốn sử dụng. Sử dụng GPU sẽ tốt hơn CPU (giảm thời gian huấn luyện từ vài giờ xuống còn vài phút)

Chúng tôi khuyến khích bạn nên lựa chọn:
Google Colab: bởi họ cung cấp GPUs miễn phí, có phiên bản mới nhất của TensorFlow
Nếu bạn có một máy tính đủ khỏe, thì bạn nên sử dụng máy tính của bạn

Dù bạn có lựa chọn như nào thì bạn cần có một phiên bản đồng bộ hóa công việc của mình</td>
  </tr>
</table>


* [Welcome video](https://www.coursera.org/learn/language-processing/lecture/cnBKV/welcome-video)

<table>
  <tr>
    <td>Hi everyone. I am excited to see you on board, and welcome to our course. I want to start our lesson with the informal discussion of who we are, and who is this course for.
Then, we will have a brief introduction to the area of Natural Language Processing. You know, it might feel a little hand wavy as any introduction actually, but I hope that after our course, you will know exactly everything that will be mentioned in our lesson now.
So, ready? Let us get started.
My name is Anna, and we have a big nice team creating the course for you.
So, we have Sergey, Alexey, Andrey, and one more Anna preparing the materials. 
I have a background on computer science and machine learning, and I'm now applying this background in natural language processing in different ways. And you know these different activities like research, teaching, and industry, give rather different perspectives to the same area.
So, for example, when you come to the industry very soon, you realize that not any paper from academia is useful in the particular settings, like large scale implementation or some noisy data or specific needs of your business. So, probably, you need to build some more simple solution but that would work nicely in your specific settings.
Okay. Now, who is this course for?
When I was thinking about what would be one word to characterize our audience,I thought that it would be the word curious. So, this course is for curious people who want to know what is inside some applications. For example, you have differently used machine translation.
Do you know how it works or dialogue agents that are so popular nowadays? What is inside there? And you know, this popularity of certain applications is couldn't bet. So, for example, for dialogue agents, we have so much hype around so that it is not that easy to distinguish what is just some beautiful words, and what is something that will really work in practice. So, hopefully, one outcome of our course for you would be the ability to distinguish between the hype and something that really works. 
Now, our course is rather in-depth.
So, I want to go with some details through several methods in NLP because these will give you the ability to distinguish the hype from the methods. 
Okay? Also, we will cover real state-of-the-art approaches both in research and production.And as I have already said, this could be rather different approaches. Now, another goal that's a little bit contradict to going in-depth would be to have a big picture of the area. So, I feel like it is really important to have some expertise like, I am given a task, what should I do with it? What approaches would work in this certain case? To have this intuition, we will try to discuss as many different settings and tasks as possible, and cover some approaches for them. And obviously, we should not only talk and you should not only listen and read about it, but you have to do some practice to get a hands-on experience. So, we are preparing materials for you for home assignments in Python for some popular NLP tasks like text classification, or duplicate detection, named entity recognition, and some others, so that you have some experience with your own hands. Also, this home tasks will help you to build the project of our course that would be a conversational chat-bot. So now, I feel like it is really important also to see what is our course not about, because NLP is so big that obviously our course cannot feed everyone's needs. So, I feel like if you only want to know some black box implementations and stock them together to build some solution, then probably, this course is not for you. Also, I think that it is a good idea to take machine learning and deep learning courses first to fill it is with some names and formulas. 
For example here, I have a quick test for you.  Do you know what is Recurrent Neural Networks? Or have you heard about likelihood maximization? Just take a moment to see how comfortable you are with these words and see whether you need to take, for example, deep learning course in our specialization first before going to this course. Also, we expect that you have some experience with Python. Probably, you don't have any experience with TensorFlow, and this is maybe okay, and then this is a good moment for you to try to go through some tutorials, and this course could be a good reason to go through them. Actually, TensorFlow has really nice tutorials, so I think that it shouldn't be a problem for you.
I hope you are still not frightened.
And I hope you are ready for our journey to the NLP.
And I want to start this journey with the survey of the main approaches.</td>
    <td>Mục đích của khóa học là để người học có thể nhận biết được những gì đã được cường điệu và những gì mà trong tầm tay chúng ta có thể làm được.

Bên cạnh việc lắng nghe và đọc bài giảng, chúng ta cần thực hành, và dự án của khóa học này sẽ là xây dựng một Chatbot

Chúng tôi mong rằng bạn đã từng làm việc với Python. Bên cạnh đó, nếu bạn chưa từng tiếp xúc với Tensorflow thì đó cũng không phải là vấn đề quá lớn. Tuy nhiên, TensorFlow thực sự rất hữu ích và chúng ta nên tìm hiểu về nó.</td>
  </tr>
</table>


* [Main approaches in NLP](https://www.coursera.org/learn/language-processing/lecture/j8kee/main-approaches-in-nlp)

<table>
  <tr>
    <td>I would say that there are three main groups of methods in NLP. One group would be about rule-based approaches. So, for example regular expressions would go to this group. Another one would be traditional machine learning. And the last one would be deep learning that has recently gained lots of popularity in NLP. In this video, I want to go through all three approaches just by example of one particular tasks, so that you get some flavor of all of them. The task could be called semantic slot filling. So, you can see the query in the bottom of the slide which says Show me flights from Boston to San Francisco on Tuesday. So you have some sequence of words, and you want to find some slots. So the slots would be destinations or departure or some date and something like that. And to fill those slots you can use different approaches. This slide is about context-free grammars. So, it is a rule-based approach. The context-free grammars show you what would be the rules to produce some words. For example, you can see that non terminal words show can produce words, show me, or can I see, or something like that. And some other words for example origin, non terminal can produce from city, and city non terminal can then produce some specific cities from a list. Now, when you have this context-free grammar, you can use it to parse your data. So you can get to the sequence and say what are the non terminals that created this certain words. So what will be the advantages and disadvantages of this approach? Well, this approach is usually done manually. So you have to write all those rules just by yourself or some linguists should come and write it for you. So obviously, this is very time consuming. Also the record of this approach would be not very nice because well, you cannot write down all the possible cities because there are so many of them and the language is so very native. Right? So, the positive thing though would be the precision of this approach. Usually, rule-based approaches have high precision but low recall. Now, another approach would be to build some machine learning system. To do that, first of all you need some training data. So you need a corpus with some markup. So here, you have a sequence of words and you know that these certain phrases have these certain texts. Right? Like origin, destination, and date. After you have your training data, you need to do some feature engineering. So you need to create features like for example is the word capitalized? Or does this word occur in some list of Cities or something like that. Then, you need to define your model. So the probabilistic model would for example produce the probabilities of your text given your words. This can be different kinds of models and we will explore a lot of them in our course. But generally, these models would have some parameters and they will depend on some features that you have just generated. And the parameters of the model should be trained. So you will need to take your train data and fit your model to this data. So you will maximize the probability of what you see, by the parameters. This way, you will fix the parameters of the model and you will be able to apply this model to the test data. For the inference, you will apply it, and you will find the most probable text for your words with some fixed parameters. Right? So this is called inference or test or deployment or something like that. So, this is just the general framework. Right? So you have some perimeters, you train them, and then you apply your model. The similar thing happens for deep learning approach. There you also have this stages but usually you do not have the stage of feature generation. So what you're doing is that you just feed your sequence of words as is to some neural network. So, I now do not go into the details of the neural network. We will have time to go into those details. I just show you the idea that you feed your words just as one hot encoders. As the vectors that have only one non zero element that corresponds to the number of this word in the vocabulary and lots of zeros. So you feed this vectors to some complicated neural network that has some complicated architecture and lots of parameters. You feed this perimeters and then you apply this network to your test data to get the text out of this model. Deep learning methods perform a really nice for many tasks in NLP. So, sometimes it feels like we forget about traditional approaches, and there are some reasons not to forget about it. Well, the first reason would be that traditional methods perform really nice for some obligations. For example for sequence labeling, we can do probabilistic modeling and we will discuss it during the week two, and we'll get a really good performance. Another reason would be that some ideas in deep learning methods are really similar to something that was happening in the area before them. So, for example, word2vec method which is actually not even deep learning but it is inspired by some neural networks, has really similar ideas as some distributional semantic methods have. And in week three of our course, we will discuss both of them. Now, another reason would be that we can sometimes use the knowledge that we had in traditional approaches to improve the models based on deep learning. For example, word alignments in machine translation and attention the Haney's in neural networks are very similar, and we will see during the week four. Deep learning methods are indeed fancy and we have lots of research publications about them in our current conferences. So, it looks like this is where the area will go in the future. So, obviously, we need to have them in our course as well. So what do we do? Well, I think that we will have both of them in parallel. So, for every task, we'll have traditional and deep learning approaches studied one by one. And this is all for this video. And in the next video we, will see what is the plan for our next week's.</td>
    <td>Có thể nói rằng chúng ta có 3 phương pháp chính trong NLP:
- tiếp cận dựa trên quy tắc
- học máy truyền thống
- học sâu

Video này sẽ giới thiệu về cả ba phương pháp thông qua bài toán điền vào chỗ trống ngữ nghĩa

Bạn có một vài từ và muốn điền thông tin còn sót lại. Để điền được những thông tin đó, bạn có thể sử dụng nhiều cách tiếp cận khác nhau. Ở đây chúng ta đang nói về văn phạm phi ngữ cảnh. Bạn có thể sử dụng văn phạm phi ngữ cảnh để phân tích dữ liệu của bạn, từ đó có thể lấy ra được các dãy từ. Mặc dù phương pháp này đem lại độ chính xác cao nhưng nó được làm hoàn toàn thủ công và tốn nhiều thời gian, nên chúng ta sẽ chuyển sang hướng tiếp cận khác.
Chúng ta sẽ dựa vào các dữ liệu huấn luyện vào xây dựng lên một hệ thống học máy. Bạn cần phải xây dựng được mô hình cho hệ thống của bạn. Mô hình xác suất được dùng để mô tả dữ liệu của bạn dựa trên các từ. Các mô hình sẽ cần đến các tham số, và các tham số này cũng cần phải được huấn luyện để mô hình có thể phù hợp với dữ liệu đầu vào. Đây là quy trình đơn giản áp dụng cho cách tiếp cận học sâu.
Ý tưởng là chúng ta sẽ dùng các one-hot vector (vector chỉ có duy nhất một phần tử khác 0) để biểu diễn các từ, sau đó đưa các vector đó vào mạng nơ ron huấn luyện, và lấy ra văn bản trả về.
Học máy đáp ứng khá tốt trong các bài toán NLP nên chúng ta thường bỏ qua các cách tiếp cận truyền thống. Tuy nhiên, chúng ta không nên bỏ qua các tiếp cận truyền thống bởi:
độ chính xác cao ngay cả đối với những bài toán phức tạp.
ý tưởng tới cho các bài toán học sâu đều xuất phát từ các cách tiếp cận truyền thống
sử dụng kiến thức từ các cách tiếp cận truyền thống có thể cải thiện mô hình học sâu</td>
  </tr>
</table>


* [Brief overview of the next weeks](https://www.coursera.org/learn/language-processing/lecture/8W5T4/brief-overview-of-the-next-weeks)

<table>
  <tr>
    <td>Hey. In this video, we will briefly discuss what will be covered during the next weeks. So, during this week we will discuss text classification tasks. So these are the tasks that are very popular in any applications. For example, you need to predict sentiment for some reviews. You need to know whether the review is positive or negative. Or, you need to filter spam in your emails or something else. So, what you do is actually you represent your text as a bag of words, you compute some nice features and you apply some machine learning algorithm to predict the class of this text. There are actually lots of practical tips that you need to know to succeed in this task. So, during this week my colleague will tell you about it. Now, the next week will be about representing text not as a bag of words but as a sequence. So, what can you do when you represent a text as a sequence of words? One task would be language modeling. So language models are about predicting the probabilities of the next words given some previous words. So, this can be used to do text generation. And this is useful in many applications. For example, if you do machine translation, you are given some sequence of words, some sentence on English and then you need to translate it, let's say to Russian, so you need to generate some Russian text and that is where you'll need language model. Now, another important task is called sequence tagging. So this is the task when you have a sequence of words and you need to predict text for each of the words in this sequence. For example, it could be part-of-speech texts so you need to know that some words are nouns, some words are verbs and so on. Another task would be to find named entities and this is really useful. For example, you can find some names of the cities and use them as features for your previous task for text classification. Now, another task which is called semantic slot filling has been just covered in our previous video. So this is about some slots. For example, you need to pass a query and you need to know that the person wants to book a table for some specific time in some specific place. And those time and place would be the slots for you. Now, we can do something even more complicated and try to understand the meaning of words or some pieces of text. How do we represent the meaning? Well, one easy way to do this would be to use vectors. So, you map all the words to some vectors. Let's say 300 dimensional vectors of some float numbers and these vectors will have really nice properties. So, similar words will have similar vectors. For example, this nice picture tells you that Cappuccino and Espresso are the same thing just because the cosine similarity between the vectors is really high. Now, we will also discuss topic models. Topic models deal with documents as a whole and they also represent them by some vectors that can tell you what are the topics in these certain documents. This is really useful when you need to, for example, describe the topics of a big data set like Wikipedia or some news flows or social networks or any other text data that you are interested in. Now, this is just another example of how those methods can be used. So, let's say that we could represent our words by vectors just by three dimensional vectors and we have them depicted here in this space. And we know the similarity between them. So, we know the distance between those blue dots. Once we know these distances, we can create a similarity graph for words. So, in the middle picture, the nodes are words, and the edges have the similarities between the nodes. Now, this graph is actually very useful. Why? Because when you have some labels for some nodes of this graph, for example, if you know that "Laugh" has the label "Funny," you can try to propagate these labels through the graph. So those words that are similar will get the same labels. For example, the word "Haha" there will also get the label "Funny" because it is similar to the word "Laugh." Okay. This can be used in many different applications and we'll cover this in week three. Now, the next one will be more advanced and this week will be about sequence to sequence tasks. Actually, nearly any task in NLP can be somehow stated as a sequence to sequence tasks. Just to give you a few examples, it would be about machine translation. So there, obviously, you have one sentence and you need to translate it to the other sentence. So, these are the two sequences. But, for example, in summarization, you have the big document as an input. This is some long sequence and you need to produce some short summary, and this is also a sequence. You get this task in speech recognition or in conversational chat-bot where you have some questions and answers. Right? All these tasks can be nicely solved with so-called encoder-decoder architecture in neural networks. Let us see just the idea. So, we're given a sentence and we have an encoder. Now we feed this sentence to the encoder. What we get is some hidden representation of the input sentence. After that, the decoder generates the output sentence. So, this is how we get our final translation, or summary, or something else. Now, during the last week of our course, we will combine all the knowledge that we have to build a dialogue system. So, dialogue systems can be different and there are at least two important types. One type would be goal-oriented agents that try to solve some particular task. For example, they can assist you in a bank, or help you with online shopping, or something like that. On the contrary, there are also conversational, entertaining, chat-bots that just wants to somehow hold a conversation with you. So, there are different types of methods to be used in these types of tasks and we will cover them in details during the last week. And the project will be about stack overflow chat-bot that tries to assist with the search. So, stay with us and we will discuss everything in many details.</td>
    <td>Trong tuần này chúng ta sẽ thảo luận về bài toán phân loại văn bản. Đây là những bài toán rất phổ biến trong mọi ứng dụng. Ví dụ, bạn cần phải dự đoán tính chất của một số nhận xét. Bạn cần phải biết liệu bài đánh giá là tích cực hay tiêu cực. Hoặc, bạn cần phải lọc thư rác trong email của mình hoặc thư khác.
Tuần tiếp theo chúng ta sẽ tìm hiểu về văn bản không phải là một tập hợp các từ mà là một chuỗi. Bài toán có thể áp dụng sẽ là mô hình hóa ngôn ngữ. Các mô hình ngôn ngữ dự đoán xác suất của các từ tiếp theo dựa trên một số từ trước đó -> có thể tạo văn bản. Ví dụ, nếu bạn muốn dịch một số câu tiếng Anh sang tiếng Nga,bạn sẽ cần mô hình ngôn ngữ. 
Bây giờ, một bài toán quan trọng khác được gọi là gắn nhãn chuỗi. Đây là bài toán khi bạn có một chuỗi các từ và bạn cần phải dự đoán từ loại cho mỗi từ trong chuỗi này. 
Một bài toán khác là tìm các thực thể. Ví dụ, bạn có thể tìm thấy một số tên của các thành phố và sử dụng chúng cho bài toán phân loại văn bản. 
Một bài toán khác được gọi là điền vào chỗ trống ngữ nghĩa đã được đề cập trong video trước của chúng tôi. Ví dụ, bạn cần phải truy vấn và bạn cần phải biết rằng người đó muốn đặt bàn trong một số thời gian cụ thể ở một số nơi cụ thể. Và thời gian và địa điểm đó sẽ là chỗ trống. 
Để có thể hiểu được ý nghĩa của 1 từ hoặc 1 văn bản, bạn có thể sử dụng vectơ. Bạn ánh xạ tất cả các từ cho một số vectơ. Bạn có thể đánh giá sự giống nhau giữa hai từ dựa tren sự giống nhau về cosin giữa các vectơ. 
Bây giờ, chúng ta cũng sẽ thảo luận về các mô hình chủ đề. Các mô hình chủ đề cũng có thể đưa các văn bản của bạn thành dạng các vector để dễ dàng tính toán.
Bây giờ, đây chỉ là một ví dụ khác mà các phương pháp trên có thể được sử dụng. Chúng ta có thể biểu diễn các từ của chúng ta bằng vector ba chiều và chúng ta có chúng được mô tả trong không gian này. Và chúng ta biết sự tương đồng giữa chúng. Vì vậy, chúng ta biết khoảng cách giữa những chấm màu xanh. Khi chúng ta biết những khoảng cách này, chúng ta có thể tạo một biểu đồ tương tự cho các từ. Ở giữa hình ảnh, các nút là các từ và các cạnh có các điểm giống nhau giữa các nút. Từ đó chúng ta có thể đánh giá được các từ đồng nghĩa với nhau.




















Phần tiếp theo sẽ nâng cao hơn là bài toán các chuỗi. Ví dụ, đó là về dịch. Ví dụ, trong tóm tắt, bạn có tài liệu lớn làm đầu vào. Đây là một chuỗi dài và bạn cần tạo ra một số tóm tắt ngắn, và đây cũng là một chuỗi. Bạn nhận nhiệm vụ này trong nhận dạng giọng nói hoặc trong trò chuyện trò chuyện bot, nơi bạn có một số câu hỏi và câu trả lời. Nhiệm vụ này có thể được giải quyết nhờ mạng nơ ron. Chúng tôi đã đưa ra một câu và chúng tôi có một bộ mã hóa. Bây giờ chúng ta đưa câu này vào bộ mã hóa. Những gì chúng tôi nhận được là một số đại diện ẩn của câu đầu vào. Sau đó, bộ giải mã tạo ra câu đầu ra. Đây là cách chúng tôi nhận được bản dịch cuối cùng của chúng tôi, hoặc tóm tắt. 
Trong tuần cuối cùng của khóa học, chúng tôi sẽ kết hợp tất cả các kiến ​​thức mà chúng tôi phải xây dựng một hệ thống đối thoại. Hệ thống đối thoại có thể khác nhau và có ít nhất hai loại quan trọng. Một loại sẽ là các tác nhân hướng mục tiêu cố gắng giải quyết một số nhiệm vụ cụ thể. Ví dụ, họ có thể giúp bạn trong một ngân hàng, hoặc giúp bạn mua sắm trực tuyến, hoặc một cái gì đó như thế. Ngược lại, cũng có những trò chuyện trò chuyện, giải trí, trò chuyện mà chỉ muốn bằng cách nào đó tổ chức một cuộc trò chuyện với bạn. Có nhiều loại phương pháp khác nhau sẽ được sử dụng trong các loại tác vụ này và chúng tôi sẽ đề cập đến chúng chi tiết trong tuần trước. Dự án sẽ là về chatbot trò chuyện cố gắng hỗ trợ tìm kiếm.</td>
  </tr>
</table>


* [ Optional linguistic knowledge in NLP](https://www.coursera.org/learn/language-processing/lecture/2fFlE/optional-linguistic-knowledge-in-nlp)

<table>
  <tr>
    <td>In this video, I want to remind you that NLP area is not only about mathematics but it also about linguistics, and it is really important to remember it. So the first slide will be about this picture that is really very popular in many introductions to NLP. But I think that we also need to briefly cover it. So let us say that we are given some sentence. There are different stages of analysis for that sentence. The first stage, which is called morphological stage,
would be about different forms of words. For example, we care about part of speech text, we care about different cases and genders and tenses. So this is everything that goes just for single words in the sentence. Then the next stage, syntactical analysis, will be about different relations between words in the sentence. For example, we can know that there are some objects and subjects and so on. Now the next stage, once we know some synthetic structures, would be about semantics. So semantics is about the meaning. So you see, we are going higher and higher in our abstraction, going from just some symbols to some meanings. And to be pragmatics would be the highest level of this abstraction. Now, one reason why we do not cover all this building blocks in many details later in our course is that you can just use some very nice log books implementations for low level stages. For example for morphological and syntactical analysis, you might try using analytical library which is a really convenient tool in Python. So please feel free to investigate it. And another thing that I wanted to mention is Stanford parser. It is a parser for synthetic analysis that provides different options and has really lots of different models built in. Now Gensim and MALLET would be about more high level abstractions. For example, you can do subclassification problems there or you can think about semantics. So you have there topic models and some word embeddings representations that we will discuss later in week three. Now, another thing which also comes from linguistic part of our area is different types of relations between the words. And linguists know really a lot about what could be that types. And this knowledge can be found in some extrinsic resources. For example, WordNet is a resource that tells you that there are, for example, some hierarchical relationships. Like, we have some fruits, and then some particular types of fruits like peach, apple, orange, and so on. So this relation would be called hyponym and hypernym. And there are also some other relationships like part and the whole. For example, you have a wheel and a car. So this type of relationship is called meronyms. Now this type of relationships can be found in the WordNet resource. Here in this slide, I have a picture of another resource, BabelNet. The BabelNet resource is multilingual, so you can find some concepts in different languages there. And what is nice, you have some relations between these concepts. So for example, I just typed in NOP there and then I have seen part of speech taking test. I clicked into this test and I could see some nearest neighbors in this space of concepts. For example, I can see that the Viterbi algorithm and Baum-Welch algorithm are somewhere close by. And after Week two of our course, you'll know that they are indeed very related to this task. So the takeaway from this slide would be to remember that there are some extrinsic resources that can be nicely used in our applications. For example, how can they be used? This is a rather complicated task. It is called reasoning, and it says that there is some story in a natural language. For example, Mary got the football, she went to the kitchen, she left the ball there. Okay, so we have some story, and now we have a question after this story, where is the football now? So to answer this question, the machine needs to somehow understand something, right. And the way that we can build this system would be based on deep learning. So you might have heard
about LSTM networks, it is a particular type of recurrent neural networks. But here, you see that you have not only the sequential transition edges in your data representation, but you have also some other edges. Those red ages, tell you about coreference. Coreference is another linguistic type of relation between the words that says like she is the same as Mary,
right? So she is just a substitute for the Mary. And for example, this and that football
is the same ball, just mentioned twice. The green think is about hypernym
relationship that I have briefly mentioned. So the football is a particular
type of the balls, right. So once we know that our words
have some relationships, we can add some additional
edges to our data structure. And after that, we can have so
called DAG-LSTM, which is dynamic acyclic graph-LSTM that
will try to utilize these edges, okay? So I'm not going now to
cover this DAG-LSTM model. I just want you to see that there is
a way to use the linguistic knowledge to our needs here and to improve
the performance of some particular question answering task, for example. In the rest of the video, I want to cover another example of
linguistic information used in the system. So this will be about syntax. So let us have just a few more details
of how syntax can be represented. Usually these are some kinds of trees. So here you can see the dependency
tree and it says that, for example, the word shot is
the main word here and it has the subject, I,
and the object elephant. And the elephant has a modifier an,
and so on. Right, so you have some
dependencies between the words. And usually you can obtain these
by some syntactic parsers. Another way to represent syntax would
be so-called constituency trees. So you can see the same sentence
in the bottom of the slide. And then you parse it from bottom to
top to get this hierarchical structure. So you know that an and elephant are
determinant and noun, respectively, and then you merge them to get a noun phrase. Also, there to merge it with a verb,
which is short and yet a verb phrase. You merge it with another subtree and
get a big verb phrase. And finally, this verb phrase plus noun
phrase, I, gives you the whole sentence. Actually, you can stop at some moment so you cannot parse the whole
structure from bottom to the top. But just say that, it is enough for
you to know that for example, in my picture is
some particular subtree. Why can it be useful? So first, it is called shallow parsing. And it used for example in named
entities recognition because a named entity is a very likely to be
a noun phrase, just altogether, right. New York City would be a nice
noun phrase in some sentence. So, it can help there but it can also help
as the whole tree in some other tasks. And the example of some of this
task would be sentiment analysis. The sentiment analysis treats
reviews as some pieces of text and tries to predict whether they are positive
or negative, or maybe neutral. So here you can see that
you have some pluses and minuses and zeros,
which stand for the sentiment. So you have your sentence, right? And then you try to parse
it with your syntax. So you get this nice subtrees that we
have just seen in the previous slide. And the idea is, that if you know
the sentiment of some particular words, for example you know that humor is good. Then you can try to merge those sentiment to produce the sentiment
of the whole phrase. Okay, so intelligent humor are both good
and they give you some good sentiment. But then when you have some not
in the sentence, you get the not good sentiment, which results in the
negative sentiment for the whole sentence. So this is rather advanced approach. It is called recursive neural networks, or dynamic acyclic graph neural networks,
and so on. Sometimes they can be useful,
but in many practical cases, it is just enough to do some more
simple classification for your work. So in the rest of this week, my colleague
will discuss classification task. For example, for
sentiment analysis in many, many details.</td>
    <td>NLP không chỉ về toán học mà còn bao gồm ngôn ngữ học, và điều này rất quan trọng cần phải ghi nhớ.
Giả sử chúng ta có một vài câu, thì chúng ta sẽ có một vài bước khác nhau để phân tích những câu nói đó.
Bước thứ nhất, đó là giai đoạn hình thái, tìm hiểu về các dạng khác nhau của từ, như thì của từ, ngữ cảnh, ... Sau đó, chúng ta sẽ phân tích cú pháp của câu - mối quan hệ giữa các từ trong câu. Chúng ta sẽ biết về các cấu trúc tổng hợp, và ngữ nghĩa của câu.
Chúng ta có thể sử dụng một vài thư viện cho những bước đầu tiên dựa vào ngôn ngữ Python. Bên cạnh đó, chúng ta cũng có thể sử dụng bộ phân tích của Stanford cho tổng hợp câu, và cuối cùng là Gensim và MALLET cho những bước sâu hơn.





















Ví dụ, bạn muốn phân lớp các vấn đề ở đây hoặc bạn có thể nghĩa đến phân tích ngữ nghĩa. Bạn có mô hình chủ đề và các từ nhúng đại diện.

Một vấn đề nữa đó là sự khác nhau giữa loại của các mối quan hệ giữa các từ. Ví dụ, chúng ta có thể sử dụng WordNet để đưa ra các mối quan hệ phân tầng.
Mối quan hệ phân tầng có thể là mối quan hệ chung-riêng, bộ phận-toàn thể,...











Ngoài ra chúng ta cũng có thể dùng BabelNet để tìm ra các khái niệm của các ngôn ngữ khác nhau. Sau khi có được mối quan hệ giữa các từ, tôi có thể thấy được từng phần của câu nói. Tôi có thể thấy được các "hàng xóm" gần nhất trong không gian khái niệm. Ví dụ như thuật toán Viterbi và thuật toán Baum-Welch có nét nào đó tương đồng với nhau.





Bài giảng này đưa ra các nguồn bên ngoài mà chúng ta có thể sử dụng trong ứng dụng của mình.
Ví dụ ta có câu: Mary có một quả bóng, cô ấy đi vào bếp và cô ấy để quả bóng ở đó. Câu hỏi là, vậy quả bóng đang ở đâu?
Để trả lời câu hỏi này, máy cần phải hiểu được một vài điều nên một trong những các có thể làm đó là xây dựng hệ thống dựa trên cơ chế học sâu. Bạn có thể sử dụng mạng LSTM (một dạng của RNN). Bên cạnh đó, bạn cũng phải phân tích sự đồng tương quan giữa hai từ, ví dụ như từ cô ấy ở đây là để ám chỉ Mary.























Chúng ta có thể dùng DAG-LSTM, là một đồ thị tuần hoàn.









Phần này tôi sẽ tập trung vào cú pháp của ngôn ngữ, thường thì sẽ được biểu diễn bởi một vài dạng cây.









































Một ví dụ nữa của phần này là phân tích ngữ nghĩa. Nó có thể phân tích từng phần của đoạn văn và dự đoán xem đó là tích cực, tiêu cực hay trung tính.










Ý tưởng là, nếu bạn đã có dữ liệu cho ngữ nghĩa của một vài từ, bạn có thể dùng nó để dự đoán cho cả đoạn đó. Bạn có thể sử dụng RNN hoặc DAGNN (recursive neural networks, dynamic acyclic graph neural networks). Những thuật toán này thường sử dụng được cho các bài toán đơn giản, và đối với những bài toán phức tạp, chúng ta cần phải phân tích chúng kĩ hơn.</td>
  </tr>
</table>


    1. How to: from plain texts to their classification

        * [Text preprocessing](https://www.coursera.org/learn/language-processing/lecture/SCd4G/text-preprocessing)

<table>
  <tr>
    <td>Hi! My name is Andre and this week, we will focus on text classification problem. 
Although, the methods that we will overview can be applied to text regression as well, but that will be easier to keep in mind text classification problem. And for the example of such problem, we can take sentiment analysis. That is the problem when you have a text of review as an input, and as an output, you have to produce the class of sentiment. 
For example, it could be two classes like positive and negative. It could be more fine grained like positive, somewhat positive, neutral, somewhat negative, and negative, and so forth. And the example of positive review is the following. "The hotel is really beautiful. Very nice and helpful service at the front desk." So we read that and we understand that is a positive review. As for the negative review, "We had problems to get the Wi-Fi working. The pool area was occupied with young party animals, so the area wasn't fun for us." So, it's easy for us to read this text and to understand whether it has positive or negative sentiment but for computer that is much more difficult. And we'll first start with text preprocessing. And the first thing we have to ask ourselves, is what is text? You can think of text as a sequence, and it can be a sequence of different things. It can be a sequence of characters, that is a very low level representation of text. You can think of it as a sequence of words or maybe more high level features like, phrases like, "I don't really like", that could be a phrase, or a named entity like, the history of museum or the museum of history. And, it could be like bigger chunks like sentences or paragraphs and so forth. Let's start with words and let's denote what word is. It seems natural to think of a text as a sequence of words and you can think of a word as a meaningful sequence of characters. So, it has some meaning and it is usually like, if we take English language for example, it is usually easy to find the boundaries of words because in English we can split up a sentence by spaces or punctuation and all that is left are words. Let's look at the example, Friends, Romans, Countrymen, lend me your ears; so it has commas, it has a semicolon and it has spaces. And if we split them those, then we will get words that are ready for further analysis like Friends, Romans, Countrymen, and so forth. It could be more difficult in German, because in German, there are compound words which are written without spaces at all. And, the longest word that is still in use is the following, you can see it on the slide and it actually stands for insurance companies which provide legal protection. So for the analysis of this text, it could be beneficial to split that compound word into separate words because every one of them actually makes sense. They're just written in such form that they don't have spaces.
 
The Japanese language is a different story. It doesn't have spaces at all, but people can still read it right. And even if you look at the example of the end of the slide, you can actually read that sentence in English but it doesn't have spaces, but that's not a problem for a human being. And the process of splitting an input text into meaningful chunks is called Tokenization, and that chunk is actually called token. You can think of a token as a useful unit for further semantic processing. It can be a word, a sentence, a paragraph or anything else. Let's look at the example of simple whitespaceTokenizer. What it does, is it splits the input sequence on white spaces, that could be a space or any other character that is not visible. And, actually, you can find that whitespaceTokenizer in Python library NLTK. And let's take an example of a text which says, this is Andrew's text, isn't it? And we split it on whitespaces. What is the problem here? So, you can see different tokens here that are left after this tokenization. The problem is that the last token, it question mark, it does have actually the same meaning as the token, it without question mark. But, if we tried to compare them, then these are different tokens. And that might be not a desirable effect. We might want to merge these two tokens because they have essentially the same meaning, as well as for the text comma, it is the same token as simply text. So let's try to also split by punctuation and for that purpose there is a tokenizer ready for you in NLTK library as well. And, this time we can get something like this. The problem with this thing, is that we have apostrophes that different tokens and we have that s, isn, and t as separate tokens as well. But the problem is, that these tokens actually don't have much meaning because it doesn't make sense to analyze that single letter t or s. It only makes sense when it is combined with apostrophe or the previous word. So, actually, we can come up with a set of rules or heuristics which you can find in TreeBanktokenizer and it actually uses the grammar rules of English language to make it tokenization that actually makes sense for further analysis. And, this is very close to perfect tokenization that we want for English language. So, Andrew and text are now different tokens and apostrophe s is left untouched as a different token and that actually makes much more sense, as well as is and n apostrophe t. Because n apostrophe t is actually, it means not like we negate the last token that we had. Let's look at Python example. You just import NLTK, you have a bunch of text and you can instantiate tokenizer like whitespace tokenizer and just called tokenize and you will have the list of tokens. You can use TreeBanktokenizer or WordPunctTokenizer that we have reviewed previously. So it's pretty easy to do tokenization in Python. 
The next thing you might want to do is token normalization. We may want the same token for different forms of the word like, we have word, wolf or wolves and this is actually the same thing, right? And we want to merge this token into a single one, wolf. We can have different examples like talk, talks or talked, then maybe it's all about the talk, and we don't really care what ending that word has. And the process of normalizing the words is called stemming or lemmatization. And stemming is a process of removing and replacing suffixes to get to the root form of the word, which is called the stem. It usually refers to heuristic that chop off suffixes or replaces them.
Another story is lemmatization. When people talk about lemmatization, they usually refer to doing things properly with the use of vocabularies and morphological analysis. This time we return the base or dictionary form of a word,
 
which is known as the lemma.
 
Let's see the examples of how it works.
 
For stemming example, there is
 
a well-known Porter's stemmer that is like the oldest stemmer for English language.
 
It has five heuristic phases of word reductions applied sequentially.
 
And let me show you the example of phase one rules.
 
They are pretty simple rules.
 
You can think of them as regular expressions.
 
So when you see the combination of characters like SSES,
 
you just replace it with SS and strip that ES at the end,
 
and it may work for word like caresses,
 
and it's successfully reduced to caress.
 
Another rule is replace IES with I.
 
And for ponies, it actually works in any way,
 
but what would you get in the result is not
 
a valid word because poni shouldn't end with I,
 
Y, and it ends with I.
 
So that is a problem.
 
But it actually works in practice,
 
and it is well-known stemmer,
 
and you can find it in an NLTK library as well.
 
Let's see other examples of how it might work.
 
For feet, it produces feet.
 
So it doesn't know anything about irregular forms.
 
For wolves, it produce wolv,
 
which is not a valid word,
 
but still it can be useful for analysis.
 
Cats become cat, and talked becomes talk.
 
So the problems are obvious.
 
It fails on the regular forms,
 
and it produces non-words.
 
But that could be not much of a problem actually.
 
In other example is lemmatization.
 
And for that purpose, you can use
 
WordNet lemmatizer that uses WordNet Database to lookup lemmas.
 
It can also be found in NLTK library,
 
and the examples are the following.
 
This time when we have a word feet,
 
is actually successfully reduced to the normalized form,
 
foot, because we have that in our database.
 
We know about words of English language and all irregular forms.
 
When you take wolves, it becomes wolf.
 
Cats become cat, and talked becomes talked, so nothing changes.
 
And the problem is lemmatizer actually doesn't really use all the forms.
 
So, for nouns, it might be like
 
the normal form or lemma could be a singular form of that noun.
 
But for verbs, that is a different story.
 
And that might actually prevents you from merging tokens that have the same meaning.
 
The takeaway is the following.
 
We need to try stemming and lemmatization,
 
and choose what works best for our task.
 
Let's look at the Python example.
 
Here, we just import NLTK library.
 
We take the bunch of text,
 
and the first thing we need to do is tokenize it.
 
And for that purpose, let's use Treebank Tokenizer that produces a list of tokens.
 
And, now, we can instantiate Porter stemmer or WordNet lemmatizer,
 
and we can call stem or lemmatize on each
 
token on our text and get the results that we have reviewed in the previous slides.
 
So it is pretty easy in Python and NLTK too.
 
So what you can do next,
 
you can further normalize those tokens.
 
And there are a bunch of different problems.
 
Let's review some of them.
 
The first problem is capital letters.
 
You can have us and us written in different forms.
 
And if both of these words are pronounced,
 
then it is safe to reduce it to the word, us.
 
And another story is when you have us and US in capital form.
 
That could be a pronoun, and a country.
 
And we need to distinguish them somehow.
 
And the problem is that,
 
if you remember that we always keep in mind that we're doing text classification,
 
and we are working on, ,
 
sentiment analysis, then it is easy to imagine
 
a review which is written with Caps Lock just like with capital letters,
 
and us could mean actually us,
 
a pronoun, but not a country.
 
So that is a very tricky part.
 
We can use heuristics for English language luckily.
 
We can lowercase the beginning of the sentence because we
 
know that every sentence starts with capital letter,
 
then it is very likely that we need to lowercase that.
 
We can also lowercase words that are seen in titles because in English language,
 
titles are written in such form that every word is capitalized,
 
so we can strip that.
 
And what else we can do is we can leave mid-sentence words as they
 
are because if they're capitalized somewhere inside the sentence,
 
maybe that means that that is a name or a named entity,
 
and we should leave it as it is.
 
Or we can go a much harder way.
 
We can use machine learning to retrieve true casing,
 
but that is out of scope of the lecture,
 
and that might be a harder problem than the original problem of sentiment analysis.
 
Another type of normalization that you can use for
 
your tokens is normalizing acronyms like ETA or E,
 
T, A, or ETA written in capital form.
 
That is the same thing.
 
That is the acronym, ETA,
 
which stands for estimated time of arrival.
 
And people might frequently use that in their reviews or chats or anywhere else.
 
And for this, we actually can write a bunch of regular expressions that
 
will capture those different representation of the same acronym,
 
and we'll normalize that.
 
But that is a pretty hard thing because you must think about
 
all the possible forms in advance and all the acronyms that you want to normalize.
 
So let's summarize.
 
We can think of text as a sequence of tokens.
 
And tokenization is a process of extracting those tokens,
 
and token is like a meaningful part,
 
a meaningful chunk of our text.
 
It could be a word,
 
a sentence or something bigger.
 
We can normalize those tokens using either stemming or lemmatization.
 
And, actually, you have to try both to decide which works best.
 
We can also normalize casing and acronyms and a bunch of different things.
 
In the next video,
 
we will transform extracted tokens into features for our model.
 </td>
    <td>Phần này sẽ tập trung vào bài toán phân loại văn bản

Phương thức mà chúng ta sẽ tìm hiểu có thể áp dụng cho hồi quy văn bản. Chúng ta có thể sử dụng phân tích ngữ nghĩa cho dạng bài toán này. 



Ví dụ với bài toán phân loại đánh giá: tích cực, tiêu cực và trung tính. 




















Chúng ta sẽ chia văn bản ra thành các câu và chia các câu thành các từ. Việc phân tách các câu có thể dựa trên dấu chấm câu. Việc phân tách các từ có thể dễ dàng với các ngôn ngữ như tiếng Anh (các từ được phân cách bởi dấu cách) hoặc khó khăn hơi như với tiếng Đức (không có dấu cách) hay như tiếng Việt (các từ có bao gồm nhiều tiếng) hoặc cũng có thể như tiếng Nhật (vừa có dấu cách vừa không có)

















Thậm chí khi bạn nhìn vào một câu tiếng anh không có dấu cách thì bạn cũng có thể đọc được câu đó.

Quá trình tách từ được gọi là Tokenization.


Mỗi một token là một đơn vị nhỏ nhất cho các bước phân tích ngữ nghĩa sau này. Nó có thể là một từ, một câu hoặc một đoạn văn.































Bạn có thể sử dụng một vài thư viện để thực hiện việc tách từ này.









Việc tiếp theo chúng ta cần phải làm đó là chuẩn hóa từ. Chúng ta có thể có những từ giống nhau nhưng ở các thì hoặc các cấu trúc khác nhau, nên chúng ta cần phải hợp các từ đó thành 1. Có hai cách tiếp cận:
- Stemming: chuẩn hóa các từ và không quan tâm đến </td>
  </tr>
</table>


        * [Feature extraction from text](https://www.coursera.org/learn/language-processing/lecture/vlmT5/feature-extraction-from-text)

<table>
  <tr>
    <td>Hi. In this lecture will transform tokens into features.
 
And the best way to do that is Bag of Words.
 
Let's count occurrences of a particular token in our text.
 
The motivation is the following.
 
We're actually looking for marker words like excellent or disappointed,
 
and we want to detect those words,
 
and make decisions based on absence or presence of that particular word,
 
and how it might work.
 
Let's take an example of three reviews like a good movie,
 
not a good movie, did not like.
 
Let's take all the possible words or tokens that we have in our documents.
 
And for each such token,
 
let's introduce a new feature or column that will correspond to that particular word.
 
So, that is a pretty huge metrics of numbers,
 
and how we translate our text into a vector in that metrics or row in that metrics.
 
So, let's take for example good movie review.
 
We have the word good,
 
which is present in our text.
 
So we put one in the column that corresponds to that word,
 
then comes word movie,
 
and we put one in the second column just
 
to show that that word is actually seen in our text.
 
We don't have any other words,
 
so all the rest are zeroes.
 
And that is a really long vector which is sparse in a sense that it has a lot of zeroes.
 
And for not a good movie,
 
it will have four ones,
 
and all the rest of zeroes and so forth.
 
This process is called text vectorization,
 
because we actually replace the text with a huge vector of numbers,
 
and each dimension of that vector corresponds to a certain token in our database.
 
You can actually see that it has some problems.
 
The first one is that we lose word order,
 
because we can actually shuffle over words,
 
and the representation on the right will stay the same.
 
And that's why it's called bag of words,
 
because it's a bag they're not ordered,
 
and so they can come up in any order.
 
And different problem is that counters are not normalized.
 
Let's solve these two problems,
 
and let's start with preserving some ordering.
 
So how can we do that?
 
Actually you can easily come to an idea that you should look at token pairs,
 
triplets, or different combinations.
 
These approach is also called as extracting n-grams.
 
One gram stands for tokens,
 
two gram stands for a token pair and so forth.
 
So let's look how it might work.
 
We have the same three reviews,
 
and now we don't only have columns that correspond to tokens,
 
but we have also columns that correspond to  token pairs.
 
And our good movie review now translates into vector,
 
which has one in a column corresponding to that token pair good movie,
 
for movie for good and so forth.
 
So, this way, we preserve some local word order,
 
and we hope that that will help us to analyze this text better.
 
The problems are obvious though.
 
This representation can have too many features,
 
because  you have 100,000 words in your database,
 
and if you try to take the pairs of those words,
 
then you can actually come up with a huge number that can exponentially
 
grow with the number of consecutive words that you want to analyze.
 
So that is a problem.
 
And to overcome that problem,
 
we can actually remove some n-grams.
 
Let's remove n-grams from features based on
 
their occurrence frequency in documents of our corpus.
 
You can actually see that for high frequency n-grams,
 
as well as for low frequency n-grams,
 
we can show why we don't need those n-grams.
 
For high frequency, if you take a text and take
 
high frequency n-grams that is seen in almost all of the documents,
 
and for English language that would be articles,
 
and preposition, and stuff like that.
 
Because they're just there for grammatical structure and they don't have much meaning.
 
These are called stop-words,
 
they won't help us to discriminate texts,
 
and we can pretty easily remove them.
 
Another story is low frequency n-grams,
 
and if you look at low frequency n-grams,
 
you actually find typos because people type with mistakes,
 
or rare n-grams that's usually not seen in any other reviews.
 
And both of them are bad for our model,
 
because if we don't remove these tokens,
 
then very likely we will overfeed,
 
because that would be a very good feature for
 
our future classifier that can just see that, okay,
 
we have a review that has a typo,
 
and we had only like two of those reviews,
 
which had those typo,
 
and it's pretty clear whether it's positive or negative.
 
So, it can learn
 
some independences that are actually not there and we don't really need them.
 
And the last one is medium frequency n-grams,
 
and those are really good n-grams,
 
because they contain n-grams that are not stop-words,
 
that are not typos and we actually look at them.
 
And, the problem is there're a lot of medium frequency n-grams.
 
And it proved to be useful to look at
 
n-gram frequency in our corpus for filtering out bad n-grams.
 
What if we can use the same frequency for ranking of medium frequency n-grams?
 
Maybe we can decide which medium frequency n-gram
 
is better and which is worse based on that frequency.
 
And the idea is the following,
 
the n-gram with smaller frequency can be more
 
discriminating because it can capture a specific issue in the review.
 
, somebody is not happy with the Wi-Fi
 
and  it says, Wi-Fi breaks often,
 
and that n-gram, Wi-Fi breaks,
 
it can not be very frequent in our database,
 
in our corpus of our documents,
 
but it can actually highlight a specific issue that we need to look closer at.
 
And to utilize that idea,
 
we will have to introduce some notions first like term frequency.
 
We will denote it as TF and that is the frequency for term t. The term is an n-gram,
 
token, or anything like that in a document d.
 
And there are different options how you can count that term frequency.
 
The first and the easiest one is binary.
 
You can actually take zero or one based on the fact
 
whether that token is absent in our text or it is present.
 
Then, a different option is to take just a raw count
 
of how many times we've seen that term in our document,
 
and let's denote that by f. Then,
 
you can take a term frequency,
 
so you can actually look at all the counts of all the terms that you have seen in
 
your document and you can normalize those counters to have a sum of one.
 
So there is a kind of a probability distribution on those tokens.
 
And for that, you take that f and divide by
 
the sum of fs for all the tokens in your document.
 
And, one more useful scheme is logarithmic normalization.
 
You take the logarithm of those counts and it actually introduces
 
a logarithmic scale for your counters and that might help you to solve the task better.
 
So that's it with term frequency.
 
We will use that in the following slides.
 
Another thing is inverse document frequency.
 
Lets denote by capital N,
 
total number of documents in our corpus,
 
and our corpus is a capital D,
 
that is the set of all our documents.
 
Now, let's look at how many documents are
 
there in that corpus that contain a specific term.
 
And that is the second line and that is the size of
 
that set that actually means the number of documents where the term appears.
 
If you think about document frequency,
 
then you would take that number of documents where
 
the term appears and divide by the total number of documents,
 
and you have a frequency of those of that term in our documents.
 
But if you want to take inverse argument frequency then you just swap
 
the up and down of that ratio and you take a logarithm of that and that thing,
 
we will call inverse document frequency.
 
So, it is just the logarithm of N over the number of documents where the term appears.
 
And using these two things,
 
IDF and term frequency,
 
we can actually come up with TF-IDF value,
 
which needs a term,
 
a document, and a corpus to be calculated.
 
And it works like the following,
 
you take the term frequency of our term T in our document d and you
 
multiplied by inverse document frequency of that term in all our documents.
 
And let's see why it actually makes sense to do something like this.
 
A high weight in TF-IDF is reached when we have
 
high term frequency in the given document and
 
the low document frequency of the term in the whole collection of documents.
 
That is precisely the idea that we wanted to follow.
 
We wanted to find
 
frequent issues in the reviews that are not so frequent in the whole data-set,
 
so specific issues and we want to highlight them.
 
Let's see how it might work.
 
We can replace counters in our bag of words representation with TF-IDF values.
 
We can also normalize the result row-wise,
 
so we normalize each row.
 
We can do, that for example,
 
by dividing by L2 norm or the sum of those numbers,
 
you can go anyway.
 
And, where we actually get in the result we have
 
not counters but some real values and let's look at this example.
 
We have a good movie,
 
two gram and it appears in two documents.
 
So in our collection it is pretty frequent, two gram.
 
That's why the value 0.17 is actually lower than 0.47 and we get 0.47 for
 
did not two gram and that actually is there because
 
that did not two gram happened only in
 
one review and that could be a specific issue and we want to highlight,
 
that we want to have a bigger value for that feature.
 
Let's look how it might work in python.
 
In Python, you can use scikit learn library and you can import TF-IDF vectorizer.
 
Let me remind you that vectorization means that we replace how we text
 
with a huge vector that has a lot of
 
zeroes but some of the values are not zeroes and those are
 
precisely the values that correspond to the tokens that is seen in our text.
 
Now, let's take an example of five different texts
 
like small movie reviews and what we do is we
 
instantiate that TF-IDF vectoriser and it has
 
some useful arguments that you can pass to it,
 
like mean DF, which stands for minimum document frequency that is
 
essentially a cutoff threshold for
 
low frequency and grams because we want to throw them away.
 
And we can actually threshold it on a maximum number of documents where
 
we've seen that token and this is done for stripping away stop words.
 
And this time, in scikit learn library,
 
we actually bust that argument as a ratio of
 
documents but not a real valued number of documents where we've seen that.
 
And the last argument is n-gram range,
 
which actually tells TF-IDF vectorizer,
 
what n-grams should be used in these back of words for representation.
 
In this scenario, they take one gram and two gram.
 
So, if we have vectorized our text we get something like this.
 
So not all possible two grams or one grams are there because some of them are filtered.
 
You can just follow,
 
just look at the reviews and see why that happened and you can
 
also see that we have real values in
 
these matters because those who actually TF-IDF values and each row is
 
normalized to have a norm of one.
 
So let's summarize, we've made actually a simple counter features in bag of words manner.
 
We replaced each text by a huge vector of counters.
 
You can add n-grams to try to preserve
 
some local ordering and we will further see
 
that it actually improves the quality of text classification.
 
You can replace counter's with
 
TF-IDF values and that usually gives you a performance boost as well.
 
In the next video,
 
we will train our first model on top of these features.
 </td>
    <td></td>
  </tr>
</table>


        * [Linear models for sentiment analysis](https://www.coursera.org/learn/language-processing/lecture/T7fNB/linear-models-for-sentiment-analysis)

<table>
  <tr>
    <td>[MUSIC] In this video, we will talk about first
text classification model on top of features that we have described. And let's continue with
the sentiment classification. We can actually take the IMDB
movie reviews dataset, that you can download,
it is freely available. It contains 25,000 positive and
25,000 negative reviews. And how did that dataset appear? You can actually look at IMDB website and
you can see that people write reviews there, and they actually also provide the
number of stars from one star to ten star. They actually rate the movie and
write the review. And if you take all those
reviews from IMDB website, you can actually use that as a dataset for text classification because you have
a text and you have a number of stars, and you can actually think
of stars as sentiment. If we have at least seven stars,
you can label it as positive sentiment. If it has at most four stars,
that means that is a bad movie for a particular person and
that is a negative sentiment. And that's how you get the dataset for
sentiment classification for free. It contains at most 30 reviews per
movie just to make it less biased for any particular movie. These dataset also provides
a 50/50 train test split so that future researchers
can use the same split and reproduce their results and
enhance the model. For evaluation, you can use accuracy and
that actually happens because we have the same number
of positive and negative reviews. So our dataset is balanced in terms
of the size of the classes so we can evaluate accuracy here. Okay, so let's start with first model. Let's takes features,
let's take bag 1-grams with TF-IDF values. And in the result,
we will have a matrix of features, 25,000 rows and 75,000 columns, and
that is a pretty huge feature matrix. And what is more, it is extremely sparse. If you look at how many 0s are there,
then you will see that 99.8% of all
values in that matrix are 0s. So that actually applies
some restrictions on the models that we can use
on top of these features. And the model that is usable for these features is logistic regression,
which works like the following. It tries to predict the probability
of a review being a positive one given the features that we gave that
model for that particular review. And the features that we use, let me remind you,
is the vector of TF-IDF values. And what you actually can do
is you can find the weight for every feature of that bag
of force representation. You can multiply each value,
each TF-IDF value by that weight, sum all of that things and pass it
through a sigmoid activation function and that's how you get
logistic regression model. And it's actually a linear
classification model and what's good about that is since it's
linear, it can handle sparse data. It's really fast to train and what's more, the weights that we get after
the training can be interpreted. And let's look at that sigmoid
graph at the bottom of the slide. If you have a linear
combination that is close to 0, that means that sigmoid will output 0.5. So the probability of a review
being positive is 0.5. So we really don't know whether
it's positive or negative. But if that linear combination in the
argument of our sigmoid function starts to become more and more positive,
so it goes further away from zero. Then you see that the probability
of a review being positive actually grows really fast. And that means that if we get the weight
of our features that are positive, then those weights will likely
correspond to the words that a positive. And if you take negative weights, they will correspond to the words that
are negative like disgusting or awful. Okay, so logistic regression can work on
these features and we can interpret it. Let's train logistic regression over
bag of 1-grams with TF-IDF values. What you can actually see is that
accuracy on test set is 88.5%. And that is a huge jump from a random
classifier which outputs 50% accuracy. Let's look at learnt features because
linear models can be interpreted. If we look at top positive weights,
then we will see such words as great, excellent, perfect, best, wonderful. So it's really cool because
the model captured that sentiment, the sentiment of those words, and
it knows nothing about English language, it knows only the examples
that we provided it with. And if we take top negative ways, then you will see words like worst,
awful, bad, waste, boring, and so forth. So these word are clearly
having negative sentiment and the model has learnt it from the examples. That is pretty cool. Let's try to make this model a little
bit better, we know how to do that. Let's introduce 2-grams to our model. And before we can move further,
we should throw away some n-grams that are not frequent, that are seen,
, less than 5 times. Because those n-grams are likely
either typos or very, like people don't say like that and
some of them do and it actually doesn't make
sense to look at those features because we will
most likely overfeed. So we want to throw that away. And if you introduce 2-grams and
that thresholding for minimum frequency, you will actually get the number of
the dimensions of our feature matrix, the following, 25,000 by 150,000. So that is a pretty huge matrix, but
we can still use linear models and it just works. Let's train logistical regression
over these bag of 1 and 2-grams with TF-IDF values. And what we actually observe is that
accuracy and test set has a bump. It has 1.5 accuracy boost. And now,
we have very close to 90% accuracy. Let's look at learnt weight. If you look at top positive weights,
then you will see that our 2-grams are actually used by our model because now
it looks at 2-grams like well worth or better than and it thinks that those
2-grams have positive sentiment. If you look on the contrary
on the top negative weights, then you will see the worst. That is another 2-gram that is now used by
our model to predict the final sentiment. You might think that, okay,
it doesn't make any sense. So the worst or worst is just the same
thing as well as well worth or just worth. So maybe it is, but that 1.5% improvement in accuracy actually was provided by addition of
those 2-grams into our model. So you can either believe it or not,
but it actually increases performance. How to make it even better? You can play around with
tokenization because in reviews, people use different stuff like emojis. They use smiles written with text. They can usually use a bunch
of exclamation marks that, a lot of exclamation marks. And you can actually look
at those sequences as, you can look at them as different tokens. And you can actually introduce
them to your model and maybe you will get a better
sentiment classification, because like a smiling face is better
than an angry face and you can use that. You should also try to normalize tokens
by applying stemming or lemmatization. You can try different models,
like SVM or Naive Bayes, or any other model that can
handle sparse features. Or another way is you can
throw bag of words away and use deep learning techniques to squeeze
the maximum accuracy from that dataset. And as for the 2016,
accuracy on this particular dataset is close to 92% and
that is a 2.5% improvement over the best model that we can get with
bag of words and 2-grams. So that might seem like not
a very good improvement, but that can actually make sense in
some tasks where you can get a lot of money even for 1% improvement, like ad
click prediction or anything like that. So let's summarize. Bag of words and simple linear models
over that features actually work. And you can add 2-grams and that is done
for free and you get a better model. The accuracy gained from deep learning
models is not mind-blowing but it is still there and
you might consider using deep learning techniques to solve the problems
of sentiment classification. In the next video,
we will look at spam filtering task, another example of task classification
that can be handled in a different way.</td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


        * [Hashing trick in spam filtering](https://www.coursera.org/learn/language-processing/lecture/XZtoZ/hashing-trick-in-spam-filtering)

<table>
  <tr>
    <td>Hi. In this video we'll talk about spam filtering task.
 
Let me remind you that when we want to use Bag-of-Words representation,
 
for every N-gram of our text,
 
we actually need find the feature index,
 
or the index of column where we will input the value,
 
 TF, IDF values into.
 
And for that purpose,
 
we need to maintain that correspondence from N-gram to feature index.
 
And usually, you use a hash map or a dictionary in Python for that.
 
Let's assume, that your data set is huge and that's where it can become a problem.
 
 we have one terabyte of text which are distributed on 10 computers,
 
and you need to vectorize each text,
 
you need to replace the text with the vector of TF IDF values.
 
You will actually have to maintain that correspondence from N-gram to feature index,
 
and that can become a problem when you
 
have 10 computers that are doing the same thing because,
 
the first problem is that,
 
that hash map can actually not fit in memory on one machine.
 
So, that means that you need a kind of a database when you store,
 
where you store these correspondence and old machines use that database,
 
that doesn't scale well.
 
And another problem is that,
 
it is difficult to synchronize that hash map because,
 
when new N-gram appears,
 
you have to introduce a new index for it,
 
and 10 machines are doing that in parallel
 
which means that they should synchronize somehow.
 
They should say that okay,
 
 I'm machine number one,
 
I found a new N-gram and I'm taking the next free index in
 
our hash map and I add that correspondence to the hash map.
 
But, that particular N-gram should be
 
converted to that feature index on all other machines as well,
 
so all other machines should somehow know,
 
that that first machine introduced a new feature index.
 
So, that can become a problem,
 
and that can actually lead to bad scaling of that workload.
 
And there is an easier way actually,
 
you can throw away hash map and you can just replace it with hashing,
 
and that means that you take N-gram,
 
you take a hash value of that N-gram and take that value modulo two to the 20,
 
or two to the 22 or any other huge number.
 
The hash is actually a function that converts an input into some number,
 
so you can give it different strings,
 
and it will output you different numbers.
 
But for some strings they can sometimes
 
output the same value and that is known as collision.
 
And hash functions have collisions but in practice we will later see that if you take it,
 
if you take that hash value modulo two to the high you rise it
 
to the high power then those collisions can be neglected.
 
You can actually, that hashing vectorizer is
 
implemented in scikit-learn and it's called hashing vectorizer obviously,
 
and it's also implemented in a Vowpal Wabbit library that we will later overview.
 
Okay, so let's take
 
spam filtering task and as you might guess that is a huge task because,
 
even if you are a medium mail server,
 
people send a lot of emails,
 
and if you have millions of users then you
 
have a terabyte of text that you need to analyze.
 
There is actually a paper on archive and
 
it actually introduces proprietary data set for spam filtering.
 
It has half a million users, three million letters,
 
and it has 40 million unique words that is seen in that letters.
 
 we map each token to index using some hash function Fi.
 
It works like the following.
 
It takes our token x, it hashes it,
 
and takes that value modulo two to the B,
 
and for B equally 22 we have four million features.
 
And that is actually a huge improvement to
 
our 40 million features that we originally had.
 
So we somehow mapped our 40 million features into four million features.
 
And thanks to the fact that hash collisions are very unlikely,
 
they are there but there are not a lot of them,
 
we can actually train the model on top of that formula and features
 
and still get the same pretty decent result.
 
So, let's look at the example of how that Hashing vectorizer works,
 
and first let me introduce some hash function for an arbitrary string.
 
We have a string S. We take the first character code of
 
that string and that is actually a number from zero to two, 55 for example.
 
And then you take the next character,
 
you multiply it by some fixed prime number.
 
Then you take the third character and multiply it by B to the two and so forth.
 
So, what do you actually obtain is an integer number,
 
and that is a hash value of your string.
 
Of course some strings can hash into the same value and that is known
 
as collision but there in practice we will not see a lot of them.
 
So let's see what we might have.
 
For this particular data set where we have three reviews,
 
good movie, not a good movie, or didn't like,
 
we take all the possible tokens that we have and
 
let's pass them through our hashing function phi,
 
and we take a pretty small B here,
 
and what we actually can get is the following: zero,
 
one, two, three and three.
 
For A and 'did' we have the same hash value but that is fine,
 
and for 'like' we have the hash value of four.
 
So, how vectorization now works?
 
And now in our columns instead of different tokens we have
 
different hash values and those are all the numbers from zero to four.
 
And let's see how our good movie now vectorizes.
 
We look at the hash value of 'good' that is zero,
 
and so we add one to the column corresponding to that value.
 
Then we take the next word which is 'movie', we hash it as well,
 
we get the value of one,
 
so we input one in the column that corresponds to that hash value.
 
And that is how we proceed with all the other reviews that we have,
 
and this actually looks pretty similar to
 
bag-of-words but now instead of tokens we have hash values.
 
Okay, let's try to train a linear model on top of these features.
 
But first, let's look at this thing.
 
Now we actually proposed a way how we
 
can squeeze the number of features that we originally had.
 
So, in a bag-of-words manner,
 
you had 40 million features and if you hash them,
 
then you have four million features.
 
And you can actually control the number of features that you have in the output
 
by adjusting that B parameter that you can find in the power of two.
 
And what that actually means is that now we can introduce a lot of tokens,
 
a lot of features, trillion features.
 
And if we hash them,
 
we still have the fixed number,
 
two to the B of features that we can analyze.
 
Let's see how it might work.
 
So phi_zero is the old hashing trick that we use.
 
We just take the hash value of our token and take that value modulo two to the B.
 
Another thing is we can actually use personalized hashing.
 
That means that we want to have a feature that says that for that particular user 'you',
 
and that particular token,
 
if you see that user and that token in the email,
 
that actually means that we want to learn
 
some personalized preference of that user in spam or non-spam e-mails.
 
And if you take a user,
 
add underscore and token,
 
and hash that new token,
 
and take that value modelo to the two to the B,
 
you have some new features.
 
And actually, in that data set,
 
if you take all pairs of user and word,
 
actually you have 16 trillion pairs,
 
and it's not possible to look at those features as
 
a bag-of-words representation because it takes 16 terabytes of data.
 
It's just not feasible.
 
But if you take the hash of those features and take it modulo two to the B,
 
you have a fixed number of features.
 
So, here we have our pipeline,
 
we have a text document.
 
We extract tokens, we add personalized tokens where we just add the prefix,
 
 user one to three,
 
underscore all the tokens that we've seen for that user,
 
we hash all those tokens and we get some sparse vector of the size two to the B. Okay.
 
Now let's see whether that hashing actually hurts our performance or not.
 
On this graph, you can see three different models.
 
The first one is denoted with black color and that is a baseline.
 
That is actually the model that was trained on
 
original tokens without any hashing just in bag-of-words manner.
 
We trained a linear model on top of bag-of-words.
 
Then the blue one is actually a hashed version where
 
you replace TfidfVectorizer with , hashingvectorizer.
 
And now you have a smaller number of features.
 
And you can see that starting from some value of b ,
 
22, you actually don't lose anything in terms of quality.
 
So if you take b equally 18,
 
then you lose some quality.
 
But if that value is huge then that is okay.
 
So it's pretty okay to use hashing if you have a lot of hash values.
 
Another thing, is there's a red curve which corresponds to personalized hashing.
 
That is the model where you introduced personalized tokens and you hashed them,
 
and you used that for linear model as well.
 
And you can see that that somehow gives you a significant boost in miss-rate.
 
So actually, on the y axis we have a miss-rate and we want to make it as low as possible.
 
Okay, so let's understand why that personalized features actually work.
 
So, the answer is pretty obvious, because they're personalized.
 
They actually capture some local user-specific preference.
 
, some users might consider newsletters as spam,
 
and that means that if you see some words that's frequently used in newsletters,
 
and then for that particular user that would be a spam as well.
 
But for the rest of the people like for the majority of them,
 
those newsletters could be okay.
 
So, if you add that personalized features,
 
you can actually understand what makes a letter a spam letter for a particular user.
 
But how will it work for new users?
 
Let's look at different users that have different number of emails in training.
 
, we take users that have 64 emails in training,
 
that means that we have a very low miss-rate for them because
 
we know really well what is a spam letter for them and what is not.
 
And if they have less and less examples,
 
it actually starts to hurt the quality of the model
 
because we have less examples of what is a spam letter for them.
 
But one surprising thing is that even for
 
users that didn't have any letters in the training set,
 
we have a higher quality than a baseline model.
 
And why does that happen?
 
Because for those users nothing changes,
 
we don't add any user-specific tokens,
 
and you can actually expect that nothing changes for them too.
 
And we get pretty close to baseline but actually it performs superior to baseline,
 
and let's find out why.
 
So, you can actually think of it in the following way.
 
It turns out that we learn better global preference when we have
 
some features that correspond to personalized preference or local use of preference.
 
Let's take the same example of people that hate newsletters.
 
There could be a small number of those people,
 
and for those people we can actually use
 
their personalized features to learn that those people hate newsletters.
 
But for majority of the people newsletters are fine.
 
And that means that having those personalized features,
 
linear models can learn then okay,
 
I will look at those personalized feature.
 
That particular person hates spam, okay,
 
hates newsletters. That is okay.
 
But for all the rest,
 
I will use the features that
 
contain the words that are seen in newsletters, like newsletter.
 
And for those people,
 
for all the rest, I will learn a better model.
 
And that what actually happens in practice and that's how
 
we can describe why this happens.
 
Another thing is, why do we deal with such huge dataset?
 
Why do we take one terabyte of data?
 
Why can't we take like a thousand of emails and just train our classifier?
 
It turns out that you can learn better models
 
using the same simple linear classifier and the same simple features,
 
but when you have more data you can learn better classifier.
 
And that can be well seen on ad click prediction.
 
There is a paper in our archive which has appropriatory dataset as well,
 
which has trillions of features,
 
billions of training examples.
 
And those people actually showed that,
 
if you sample your dataset with ,
 
you take a one percent sample or a 10 percent sample of a huge terabytes dataset,
 
then it actually hurts your model.
 
It hurts your model in terms of area under ROC curve.
 
And you can see that it hurts it with any sampling rate you take.
 
And you may think that that difference in the third digit in auROC actually makes sense.
 
You may think that,
 
okay, that is not that much,
 
why do I need to bother with that one terabyte dataset?
 
But if you are talking about ad click prediction,
 
that means that any improvement in
 
that ad click prediction can actually lead to millions of dollars of revenue.
 
So people actually want to squeeze the maximum they can from those models.
 
At last I want to overview Vowpal Wabbit library,
 
that is a well-known machine learning library that is used for training linear models.
 
It uses feature hashing that we have described previously, internally.
 
It has lots of different features.
 
And it's really fast and it scales very well.
 
And what's more wonderful is that as an input to this library,
 
you can give your raw text and it will convert it,
 
it will tokenize that text on white spaces,
 
it will take the hash value from each token,
 
and it will use that hash values internally for a hash vectorization.
 
But you can also say that you want to pass
 
your features there when you already know the hash value,
 
and you can also do that,
 
you say like 13 colon and some real value of number.
 
That means that in the column that corresponds to hash value 13,
 
you will have those value.
 
Okay, let's summarize.
 
We've taken a look on a different application,
 
particularly spam filtering that uses feature hashing.
 
And thanks to hashing,
 
you can hash a lot of features, trillion features.
 
And you can actually add personalized features,
 
and that is a really nice trick to further boost the performance of your model.
 
Linear models over bag-of-words scale well for production that is a well-known thing,
 
and that's why we actually overview them because most
 
likely you will have to implement linear model as a baseline when you work somewhere,
 
in some corporation or anywhere else.
 
In the next video,
 
we will take a look at text classification problem using deep learning techniques.
</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/HudsR/classical-text-mining)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


        * [Getting started with practical assignments](https://www.coursera.org/learn/language-processing/supplement/27CC6/getting-started-with-practical-assignments)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


        * [Predict tags on StackOverFlow with linear models](https://www.coursera.org/learn/language-processing/programming/ASMMZ/predict-tags-on-stackoverflow-with-linear-models)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


    2. Simple deep learning for text classification

        * [Neural networks for words](https://www.coursera.org/learn/language-processing/lecture/UqVpR/neural-networks-for-words)

<table>
  <tr>
    <td>Hi. In this video,
 
we will apply neural networks for text.
 
And let's first remember, what is text?
 
You can think of it as a sequence of characters,
 
words or anything else.
 
And in this video,
 
we will continue to think of text as a sequence of words or tokens.
 
And let's remember how bag of words works.
 
You have every word and forever distinct word that you have in your dataset,
 
you have a feature column.
 
And you actually effectively vectorizing
 
each word with one-hot-encoded vector that is a huge vector of
 
zeros that has only one non-zero value
 
which is in the column corresponding to that particular word.
 
So in this example,
 
we have very, good, and movie,
 
and all of them are vectorized independently.
 
And in this setting,
 
you actually for real world problems,
 
you have like hundreds of thousands of columns.
 
And how do we get to bag of words representation?
 
You can actually see that we can sum up all those values, all those vectors,
 
and we come up with a bag of
 
words vectorization that now corresponds to very, good, movie.
 
And so, it could be good to think about bag of
 
words representation as a sum of
 
sparse one-hot-encoded vectors corresponding to each particular word.
 
Okay, let's move to neural network way.
 
And opposite to the sparse way that we've seen in bag of words,
 
in neural networks, we usually like dense representation.
 
And that means that we can replace each word by a dense vector that is much shorter.
 
It can have 300 values,
 
and now it has any real valued items in those vectors.
 
And an example of such vectors is word2vec embeddings,
 
that are pretrained embeddings that are done in an unsupervised manner.
 
And we will actually dive into details on word2vec in the next two weeks.
 
But, all we have to know right now is that,
 
word2vec vectors have a nice property.
 
Words that have similar context in terms of neighboring words,
 
they tend to have vectors that are collinear,
 
that actually point to roughly the same direction.
 
And that is a very nice property that we will further use.
 
Okay, so, now we can replace each word with
 
a dense vector of 300 real values. What do we do next?
 
How can we come up with a feature descriptor for the whole text?
 
Actually, we can use the same manner as we used for bag of words.
 
We can just dig the sum of those vectors and we have
 
a representation based on word2vec embeddings for the whole text, like very good movie.
 
And, that's some of word2vec vectors actually works in practice.
 
It can give you a great baseline descriptor,
 
a baseline features for your classifier and that can actually work pretty well.
 
Another approach is doing a neural network over these embeddings.
 
Let's look at two examples.
 
We have a sentence "cat sitting there",
 
or "dog resting here",
 
and for each word,
 
we take a row that actually represents a word2vec embedding of length,  300.
 
And now we want to apply neural network here somehow.
 
And, let's first think about the following thing,
 
how do we make use of 2-grams using this representation?
 
Because, when you had bag of word representation for each particular 2-gram,
 
you had a different column,
 
and you had a very long sparse factor for all possible 2-grams.
 
But here, we don't have word2vec embeddings for token pairs,
 
we actually have word2vec embeddings only for each particular word.
 
So, how can we analyze 2-grams here?
 
Actually, it turns out that we can look at the pairs of those embedding vectors,
 
and you can think of it as a sliding window.
 
So, here in green border,
 
we have first two words,
 
and we take their word embeddings,
 
and we want to take all those values and
 
we want to analyze them somehow with neural network.
 
And for that purpose, we can actually use a convolutional filter that has the same size,
 
that has some numbers.
 
And if you take the values that are pretty close to the values that
 
correspond to "cats sitting" that means that when you convolve with that filter,
 
the 2-gram that is "cat sitting",
 
you will have a high activation just because the convolutional filter
 
is very similar to the word embeddings of these pair of words.
 
And, okay. So, now we know how we can analyze 2-grams in our text.
 
We just convolve the word vectors that are near.
 
But why is it better than bag of words?
 
In bag of words manner,
 
for each particular 2-gram,
 
we had a different column.
 
And here, we have to come off with a lot of convolutional filters that will learn
 
that representation of 2-grams and we'll be able to analyze 2-grams as well.
 
Why is it better than? It turns out that using a good property of word2vec embeddings,
 
which is the following, that similar words,
 
similar in terms of the context that they are seen in,
 
similar words, they are similar in terms of cosine distance.
 
And a cosine distance is similar to dot product.
 
And its product is actually a convolution that we're doing.
 
So, that means that if you take a different sentence like "dog resting here",
 
you can actually find that cat and dog have
 
similar representations in word2vec just because they're seen in
 
the same context like my dog ran away or my dog ate my homework.
 
And you can replace dog with cats and that would be a frequent sentence as well.
 
So, why convolutional filter is better?
 
Because, you can take an n-gram dog resting,
 
and thanks to the fact that those values are
 
pretty similar to the values of 2-gram cat sitting.
 
That means that when you convolve it with the same convolutional filter,
 
you will have a high activation value as well.
 
So, it turns out that if we have good embedding of all of vectors,
 
then using convolutions, we can actually look at more high-level meaning of the two gram.
 
It's not just cat sitting, or dog resting,
 
or cat resting, or dog sitting,
 
it actually animals sitting,
 
and that is the meaning of that 2-gram that we can learn with our convolutional filter.
 
So, this is pretty cool.
 
Now we've done neural columns for all possible 2-grams,
 
you just need to look at the pairs of word embeddings and learn
 
convolutional filters that will learn some meaningful features.
 
Okay. So, you can see that,
 
that can be easily extended to three-grams,
 
three-grams and any other n-gram.
 
And contrary to a bag-of-words representation,
 
your feature metrics won't explode,
 
because your feature metrics is actually fixed.
 
All you change is the size of the filter,
 
with which you do convolution and that is a pretty easy operation to do.
 
You can also see that just like in convolutional neural networks,
 
one filter is not enough.
 
You need to track many n-grams,
 
you need to track many different meanings of those two,
 
three grams and that's why you need a lot of convolutional filters.
 
And these filters are called 1D convolutions because
 
we actually slide the window only in one direction.
 
Contrary to ,
 
image where we slide that window both in two directions.
 
And let's see how that sliding window actually works.
 
We have an input sequence,
 
cat sitting there or here,
 
we have for each word a word to back
 
representation and we have that sliding window of size three.
 
And let's add some padding so that the size of
 
the output is the same as the size of the input.
 
Let's convolve the first patch that we got from these metrics and  we get a 0.1,
 
then 0.3, minus_0.2, 0.7 and minus_0.4.
 
And, what you actually see here is that we slide
 
that window only in one direction and that direction is actually time.
 
You can think about the sequence of words that
 
happen in time and that words occur on time axis.
 
Okay, so what do we do with these numbers now?
 
The bad property is that we have the same number
 
of outputs and it is equal to the number of inputs.
 
That means that if you have variable length of symptoms,
 
then you have variable number of features.
 
And we don't want that because we don't know what to do with that.
 
So, let's assume that just like in a bag-of-words manner,
 
we can actually lose the ordering of the words.
 
That means that we don't really care where we've seen a two-gram,
 
meaning animal sitting that we actually try to find with these convolutional filter.
 
We don't care where it occurred,
 
in the beginning of the sentence or at the end.
 
The only thing we care is whether that combination was actually in the text or not.
 
And if you assume that,
 
then all you can do is you can actually take
 
the maximum activation that you got with
 
this convolutional filter going through the whole text,
 
and you take that value as the result of your convolution,
 
and that is actually called Maximum puling over time.
 
Just like in images,
 
they have maximum puling,
 
here we apply it over time.
 
So, what we've done,
 
we're taking an input sequence,
 
we've proposed to take
 
a convolutional window size of three by the number of variables and embedding,
 
and we convolve with that filter sliding in
 
one direction and then we take the maximum activation and that is our output.
 
Okay, let's come to the final model.
 
The final architecture might look like this.
 
We can use the filters of size three, four and five,
 
so that we can capture the information about three,
 
four and five grams,
 
and for each n-gram we will learn 100 filters.
 
That means that effectively we have 300 outputs.
 
And let's look at the image.
 
We have an input sequence and  that for
 
the red window that corresponds to some convolutional filter,
 
the maximum activation was 0.7 and we have it in the output.
 
For the other filter size which is in green and  it is for two-grams,
 
if we convolve it throughout the whole sentence,
 
then the maximum value that we've seen is minus_0.7 and we add it to the output.
 
And this way using different filters of different size we have 300 outputs.
 
Okay, so what do we do with that vector?
 
That vector is actually a kind of embedding of our input sequence
 
and we've proposed a way how we can
 
convert our input sequence into a vector of fixed size.
 
What do we do next is an obvious thing.
 
We just apply some more dense layers and we actually
 
apply multi-layer positron on top of
 
those 300 features and train it for any task we want.
 
It can be either classification or regression or anything else.
 
Okay, so let's compare the quality of these model with
 
bag-of-words approach that is classical.
 
Actually, there is a link to the paper where they've done those experiments,
 
and they have a customer reviews dataset and they compared
 
their model with naive buyers on top of one and two grams.
 
And those classical model gave 86.3 accuracy.
 
And if you use
 
these proposed 1D convolutions architecture with MLP on top of those features,
 
then you get whopping 3.8 bump in
 
the accuracy and it gives you almost 90 percent accuracy.
 
And that is pretty cool because we just apply neural networks,
 
we propose how we can embed our words and we
 
can use a lot of unsupervised text for learning of those embeddings
 
and we actually proposed how we can analyze two-grams or
 
three-grams using convolutions and that are all pretty fast operations.
 
So it works even faster than bag-of-words.
 
And it works better so,
 
this is pretty cool.
 
Okay let's, summarize.
 
You can just average pre-trained word2vec embeddings for your text.
 
So you split your text into tokens.
 
For each token you take an embedding vector and you just sum them up.
 
So, that is a baseline model and it can actually work pretty well.
 
Another approach which is a little bit better
 
is to use 1D convolutions that we have described.
 
And this way you train neural network end to end.
 
So, we have an input sequence and you have a result that you want to predict,
 
and you use back propagation and train all those convolutions,
 
to train the specific features that this neural network needs to classify your sentence.
 
In the next video we will continue to apply convolutions to text.
</td>
    <td></td>
  </tr>
</table>


        * [Neural networks for characters](https://www.coursera.org/learn/language-processing/lecture/md0NM/neural-networks-for-characters)

<table>
  <tr>
    <td>Hi, in this video,
we'll go deeper with text. Let me remind you that you can think of
text as a sequence of words, or phrases, or sentences. Or, like in this video, we'll think
of text as a sequence of characters. How can we treat a text as
a sequence of characters? Lets take an example phrase cat runs, and it has underscores which
means white space and it is easy to talkanize
our text into characters. And then we can embed each characters
into vector of length  70, which has in a one hot-encoding manner. So, our alphabet is not that huge and our special characters are not that,
their number is not that huge as well, so this one hot-encoded vector will be
sparse, but it will not be that long. Okay, so what do we do next? We have some numbers now and
it looks like a sequence of those numbers. Let's start with character n-grams. It seems that we need to just like
when we were processing words in the setting when we have characters,
n-grams still make sense. And we can do that with 1D convolutions. That means that we take that C character. And we use padding here,
that white space on the left. And we take a convolution and
we get some result. And we move that window,
get another value. And we do it all the way to
the end of the sequence and get some values there as well. So this is 1D convolution because we slide
the window only in one direction, time. Okay, we can have a different kernel,
a different convolutional kernel, and we will have different values. We can take 1000 of those kernels and
we'll have 1000 filters in the result. But what's next? If you remember how we do in convolutional
networks, we usually add convolution followed by pooling, then again
convolution and pooling and so forth. So let's add pooling here. Let's see how pooling is applied. We take, let me remind you
that it works on filter level. It takes the values that are neighboring
values, and it takes the maximum of them. This time, it's 0.8. Then we move that window
with a stride of two, and we take the maximum of
those values as well. And we do it all to the end, and we do
that for all the filters we have, and this is our pooling output. Why do we need pooling? It actually introduces a little bit of position invariance
to our character n-grams. So, if that character n-gram slides
like one character to the left, to the right, there is a high
chance that thanks to pooling, the activation that we will have in
that pooling output will stay the same. Okay, and as you remember, we continue to apply convolutions
followed by pooling and so forth. So let's take that previous
pooling output, and let's apply 1D convolutions as well. So we get some filter outputs, and
we can work with those values. And what we do next is we add pooling,
and pooling works just the same. We take neighboring two values,
we take the maximum of that, then we move that sliding window, that
green window with a stride of two, and we have a different value. So that's how we applied convolution and
pooling again. And notice that our length of our like
feature representation actually decreases. That means that our receptive field
actually increases and we look at more and more characters in our input when we make
decision about activation on a deep level. We can continue to do that and
we can actually do that six times and that's how we get to
our final architecture. Our final architecture looks like this. We take first 1,000 characters of text and
in certain datasets that makes sense. It doesn't make sense to
like to read the whole text. Maybe 1,000 characters will be enough. Then we apply 1D convolution
plus max pooling 6 times, and we use the following kernel width: 7,7 and
all the rest are 3s. And we use 1,000 filters at each step. That means that after applying
that procedure 6 times, we get a matrix of features
of size 1,000 by 34. And what you can do what
those features now, you can actually apply a multi-layer
perceptron for your task. It can be regression classification or
any other laws you like. Let's see how it works on
experimental datasets. All these datasets are either
a categorization, like news datasets, or a sentiment analysis like Yelp reviews,
or Amazon reviews. And we have two categories
of these datasets. The first one, the red one,
are smaller datasets, and they contain, at most,
600,000 training samples. And we have bigger datasets that
contains millions of samples. So let's compare our models on
these two types of datasets. The first table that you see contains
errors on test set for classical models. For classical models like bag of worths or bag of worths with TFIDF with
linear model on top of that. Or you can replace tokens with n-grams and
to do the same thing. And as you can see, on small datasets, which are in red border here,
you can see that our error is the least when we use n-grams
with TFIDF, most of the time. So it tells us that if you have a small
training set then it makes sense to use classical approaches. But if your dataset grows and you have millions of examples, then maybe
you can learn some deeper representations. And that is the second table. It contains errors and
test sets for the same datasets. And here you can see LSTM and our convolutional architecture
that we have overviewed. And you can see that our architecture
actually beats LSDM on huge datasets. This gain sometimes is not that huge but
it is actually very surprising. And you can see that these deep
approaches work significantly better than classical approaches.  for
Amazon reviews which is the last column, you've got degrees in error
from roughly 8% to like 5%. So, this is pretty cool. So what we learned from this that,
we've learned the following, that deep models work better for
large datasets, and it makes sense to make all that huge
architectures when you have huge datasets. Okay, so let me summarize. You can use convolutional networks
on not only on top of words but also on top of characters. You can tweak,
text as a sequence of characters. This is called learning
from scratch in literature. It works best for large datasets where it beats classical
approaches like bag of words. And surprisingly, sometimes it even
beats LSTM that works on word level. So what you've done is you've
come to the character level and learned some deeper representations and you don't tell the system or the model
where the words are and it works better. So this is pretty cool. So this video concludes our first week. And I wish you good luck
in the following weeks. [MUSIC]
</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/20I1Q/simple-neural-networks-for-text)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


2. [Week 2](https://www.coursera.org/learn/language-processing/home/week/2)

    3. Language modeling: it's all about counting!

* [Count! N-gram language models](https://www.coursera.org/learn/language-processing/lecture/IdJFl/count-n-gram-language-models)

<table>
  <tr>
    <td>[SOUND] Hi, everyone. You are very welcome to
week two of our NLP course. And this week is about
very core NLP tasks. So we are going to speak about
language models first, and then about some models that work
with sequences of words, for example, part-of-speech tagging or
named-entity recognition. All those tasks are building blocks for
NLP applications. And they're very, very useful. So first thing's first. Let's start with language models. Imagine you see some beginning
of a sentence, like This is the. How would you continue it? Probably, as a human,
you know that This is how sounds nice, or This is did sounds not nice. You have some intuition. So how do you know this? Well, you have written books. You have seen some texts. So that's obvious for you. Can I build similar intuition for
computers? Well, we can try. So we can try to estimate probabilities of
the next words, given the previous words. But to do this, first of all,
we need some data. So let us get some toy corpus. This is a nice toy corpus about
the house that Jack built. And let us try to use it to estimate the
probability of house, given This is the. So there are four
interesting fragments here. And only one of them is
exactly what we need. This is the house. So it means that the probability
will be one 1 of 4. By c here, I denote the count. So this the count of This is the house,
or any other pieces of text. And these pieces of text are n-grams. n-gram is a sequence of n words. So we can speak about 4-grams here. We can also speak about unigrams,
bigrams, trigrams, etc. And we can try to choose the best n,
and we will speak about it later. But for now, what about bigrams? Can you imagine what happens for
bigrams, for example, how to estimate probability of Jack,
given built? Okay, so
we can count all different bigrams here, like that Jack, that lay, etc., and
say that only four of them are that Jack. It means that the probability
should be 4 divided by 10. So what's next? We can count some probabilities. We can estimate them from data. Well, why do we need this? How can we use this? Actually, we need this everywhere. So to begin with,
let's discuss this Smart Reply technology. This is a technology by Google. You can get some email, and
it tries to suggest some automatic reply. So for example, it can suggest
that you should say thank you. How does this happen? Well, this is some text generation, right? This is some language model. And we will speak about this later,
in many, many details, during week four. So also, there are some other
applications, like machine translation or speech recognition. In all of these applications, you try to
generate some text from some other data. It means that you want to
evaluate probabilities of text, probabilities of long sequences. Like here, can we evaluate
the probability of This is the house, or the probability of a long,
long sequence of 100 words? Well, it can be complicated
because maybe the whole sequence never occurs in the data. So we can count something, but we need somehow to deal with
small pieces of this sequence, right? So let's do some math to understand how to
deal with small pieces of this sequence. So here, this is our sequence of keywords. And we would like to
estimate this probability. And we can apply chain rule,
which means that we take the probability of the first word, and then condition
the next word on this word, and so on. So that's already better. But what about this last term here? It's still kind of complicated because the
prefix, the condition, there is too long. So can we get rid of it? Yes, we can. [LAUGH] So actually, Markov assumption says you shouldn't
care about all the history. You should just forget it. You should just take the last n terms and condition on them, or
to be correct, last n-1 terms. So this is where they
introduce assumption, because not everything in
the text is connected. And this is definitely very helpful for us because now we have some chance
to estimate these probabilities. So here, what happens for
n = 2, for bigram model? You can recognize that we already
know how to estimate all those small probabilities in the right-hand side,
which means we can solve our task. So for a toy corpus again,
we can estimate the probabilities. And that's what we get. Is it clear for now? I hope it is. But I want you to think about
if everything is nice here. Are we done? Well, I see at least two problems here. And I'm going to describe both of them,
and we will try to fix them. Actually, it's super easy to fix them. So first,
let's look into the first word here and the probability of this first word. So the first word can be This or That in our toy corpus. But it can never be malt or
something else. Well, maybe we should use this. So maybe we should not
spread the probability among all possible words
in the vocabulary. But we should just stick to those that
are likely to be in the first spot. And we can do this. Let us just condition our first
words on a fake start token. So let's add this fake start token in the beginning. And then we will get the probability = 1/2 for the first place, right, because we have either the This or That. So it can be helpful. What else? Actually, there is another problem here. So how do you think is this probability normalized across all different sequences of different length? Well, it's not good. So here, we have the probabilities of short sequences, of the sequences of length 1 = 1, and then all the sequences of length 2 also summing to the probability = 1. But that's not what we wanted. We wanted to have one distribution over all sequences. First, I will show how to fix you, I will show you how to fix that, like this. And then we will discuss why it helps. So let us add the fake token in the end, as we did in the beginning. And let us have this probability of the end token, given the last term. Okay, easy. Now, why does that help? So imagine some generative process. Imagine how this model generate next words. I'm going to show you the example, how it generates different sequences. And hopefully, we will see that all the sequences will fall into one big probability mass. They will spread this probability mass among them. So this is some toy corpus again. We want to evaluate this probability. And this is untouched probability mass, yet. Let's cut it, so we can go for dog or for cat. Let's go for cat. Now we can decide whether we want to go for tiger or for dog, or if we want to terminate. And this is super important. So now the model has an option to terminate. That's not what it could do without the fake end token. So this is the place where things could go wrong if we did not add this fake token. So okay, now we decide to pick something we can split further and pick and split and pick. And this is exactly the probability of this sequence that we want you to evaluate. But what's more important, you can imagine that you can also split all the other areas in different parts. And then all of them will fit into this area, right? So all the sequences of different lengths altogether will give the probability mass equal to 1, which means that it is correctly a normalized probability. Congratulations, here we are. So just to summarize, we could introduce bigram language model that splits, that factorizes the probability in two terms. And we could learn how to evaluate these terms just from data. So you can see two formulas here, in the bottom of the slide. And let it a moment to see that they are the same. So when you normalize your count of n-grams, you can either think about it as counting n-1 grams. Or you can think about it as counting n-1 gram + different continuations, all possible options, and sum over all those possible options. Okay, hope you could see that it is really about counting. And in the next videos, we will continue to study, in more details, how to train this model, how to test this model, and what other problems we have here. Thank you. 
</td>
    <td></td>
  </tr>
</table>


* [Perplexity: is our model surprised with a real text?](https://www.coursera.org/learn/language-processing/lecture/hw9ZI/perplexity-is-our-model-surprised-with-a-real-text)

<table>
  <tr>
    <td>Hey, and welcome back. This is what you have already seen in the end of our previous video. So just to remind, we have some sequences and we are going to predict the probabilities of these sequences. So we learnt that with bigger language model, you can factorize your probability into some terms. So these are the probabilities of the next word, given the previous words. Now, take a moment to see whether everything is okay with the indices on this slide. Well, you can notice that i can be equal to 0 or to k plus 1, and it goes out of range of our sequence. But that's okay because if you remember our previous video, we discussed that we should have some fake tokens in the beginning of the sequence and in the end of the sequence. So this iequal to 0 and to k plus 1 will be exactly these fake tokens. So everything good here. Let us move forward. This is just a generalization. This is n-gram language model. So the only difference here is that the history gets longer. So we condition not only on the previous words but on the whole sequence of n minus 1 previous words. So just take a note to these denotions here. This is just a brief way to show that we have a sequence of n minus one words. Great. We have some intuition how to estimate these probabilities. So you remember that we can just count some n-grams and normalize these counts. But, now, I want to give you some intuition, not only just intuition but mathematical justification. Well, we have some probabilistic model, and we have some data. And we want to learn the parameters of this model. What do we do in this case? So what you do is likelihood maximization by W train, and you note here my train data. So this is just a concatenation off all the training sequences that I have, giving a total of big M tokens. Now, I take the logarithm of this probability because it is easier to optimize the sum of logarithms, rather than the product of probabilities. And I just write down the probability of my data according to my model. Okay? So if I'm not too lazy, I would take the derivatives of this likelihood, and I would also think about constraints, such as normalization and non-negativity of my parameters. And I will derive to exactly these formulas that you will see in the bottom of this slide. So these counts and normalization of these counts have mathematical justification, which is likelihood maximization. So this is just the likelihood maximization estimates. Awesome. We now can train our language model. Now, can we show some example how it works? This is a model trained on Shakespeare corpus. So you can see that unigram model and bigram model give something meaningful, and 3-gram model and 4-gram model are probably even better. So you can see that the model actually generates some text, which resembles Shakespeare. Now, I have a question for you. How would you choose the best n here? Do you have any intuition or maybe the procedure to find the best n for your model? Well, for this case, I would say that 5-gram models usually are the best for language modeling, but it is really, really dependent on your data and on your certain task. So the general question is how do we decide which model is better? How do we evaluate and compare our models? So one way to go is to do extrinsic evaluation. So, for example, we can have some machine translation system or speech recognition system, any final application, and we can measure the quality of this application. This is a good way, but sometimes we do not have time or resources to build the whole application. Okay? So we want also to have some intrinsic evaluation, which means just to evaluate the language model itself. And one way that people use all the time is called perplexity. It is called holdout perplexity. Why? Because we have some data, and usually, we held out some data to compute perplexity later. So this is holdout data. This is just other words to say that we need some transplit and test split. So what is perplexity? Well, you know what is the likelihood. So, here, I just write down the likelihood for my test data, and perplexity is super similar. So perplexity has just likelihood in the denominator. You can be curious why exactly this formula. Well it is really related to entropy, but we are not going into details right now. So the thing that we need to know is that the lower perplexity is, the better. Why? Because the greater likelihood is, the better. So the likelihood shows whether our model is surprised with our text or not, whether our model predicts exactly the same test data that we have in real life. So perplexity has also this intuition. And, remember, the lower perplexity, the better. Let us try to compute perplexity for some small toy data. So this is some toy train corpus and toy test corpus. What is the perplexity here? Well, we shall start with computing probabilities of our model. So I compute some probability, and I get zero. It means that the probability of the whole test data is also zero, and the perplexity is infinite. And that's definitely not what we like. How can I fix that? What can we do with it? Well, there is actually a very simple way to fix that. So let us say that we have some vocabulary. Actually, that we build some vocabulary in beforehand, just by some frequencies, or we just take it from somewhere. And after that, we substitute all out of vocabulary tokens for train and for test sets for a special <UNK> token. Okay. So then we compute our probabilities as usual for all vocabulary tokens and for the <UNK> token because we also see this <UNK> token in the training data. Right? And this is what we can use because now, when we see our test data, we see they're only vocabulary tokens and <UNK> token, and we compute probabilities for all of them, and that's okay. Now, imagine we have no out of vocabulary words. We could fix that. Let's try to compute perplexity again. So this is the toy data. What is the perplexity? The probability of some tokens is still zero because we do not see this bigram in our train data, which means the whole probability is zero. The perplexity is infinite, and this is again not what we like. So for this case, we need to use some smoothing techniques. And this is exactly what our next video is about.</td>
    <td></td>
  </tr>
</table>


* [Smoothing: what if we see new n-grams?](https://www.coursera.org/learn/language-processing/lecture/EHWHb/smoothing-what-if-we-see-new-n-grams)

<table>
  <tr>
    <td>Hey, do you remember the motivation from our previous video? So the language is really variative. It means that if we train a model on a train data, it is very likely that whether we apply this model to the test data, we will get some zeros. For example, some bigrams will not occur in the test data. So, what can we do about those zeros? Could we probably just substitute them by ones? Well, actually, we cannot do this. And the reason is that in this way, you will not get a correct probability distribution, so it will not be normalized into one. Instead, we can do another simple thing. We can add one to all the accounts, even those that are not zeros. And then, we will add V, the number of words in our vocabulary, to the denominator. In this way, we will get correct probability distribution and it will have no zeroes. Now, what we have just done, so the idea is very simple. We need to somehow pull the probability mass from the frequent n-grams to infrequent ones. And this is actually the only idea about all smoothing techniques. So, in the rest of the video, we will discover what is the best way to pull the probability mass from frequent n-grams to infrequent n-grams. One rather simple approach as well would be to add not one but some k. And we can tune this constant using our test data. It will be called, Add-k smoothing. All these approaches are sometimes called Laplacian smoothing, which just may be the easiest and the most popular smoothing. Let us try to see something more complicated. So, sometimes we would like to use longer n-grams. It would be nice to use them but we might have not enough data to do this. So, the idea is, what if we try to use longer n-grams first, and then, if we have not enough data to estimate the counts for them, we will become not that greedy and go for shorter n-grams? Katz backoff is an implementation of this idea. So, let us start for example, with a five gram language model. If the counter is greater than zero, then awesome, go for it. If it's not greater than zero, then let us be not that greedy and go for a full gram language model. And again, if the counter is greater than zero, then we go for it, else we go to trigram language model. So that is simple but I have a question for you. Why do we have some alphas there and also tilde near the B in the if branch. The reason is, is that we still need to care about the probabilities. So those alpha constant is the discount that makes sure that the probability of all sequences will sum into one in our model. The same idea can be implemented in a different way. So, the Interpolation smoothing says that, let us just have the mixture of all these n-gram models for different end. For example, we will have unigram, bigram and trigram language models and we will weight them with some lambda coefficients and this lambda coefficients will sum into one, so we will still get a normal probability. And how can we find this lambdas? Well, we can just tune them using some test or some development set, if we are afraid to get or fit it. Optionally, those lambdas can also depend on some context in their more sophisticated schemes. Okay. We are doing great and we have just two methods left. The first one, is called Absolute discounting. Just to recap, the motivation for all our methods in this video is to pull the probability mass from frequent n-grams to infrequent n-grams. So, to what extent should we pull this probability mass? The answer for this question can be given by a nice experiment which was held in 1991. Let us stick to bigrams for now, and let us see that if you count the number of bigrams in your training data and after that, you count the average number of the same bigrams in the test data. Those numbers are really correlated. So, you can see that if you're just subtract 0.75 from your train data counts, you will get very good estimates for your test data and this is a little bit magical property. So, this is just a property of the language that we can try to use. The way that we use it, is let us subtract this D which is 0.75 or maybe, which is tuned using some test data that is subtracted from our counts to model the probability of our frequent n-grams. So, this is how we pull the mass and 0.75 is this extent of pull. Now, to give the probability to infrequent terms, we are using here unigram distribution. So in the right hand side, you'll see some weight, that makes sure that normalization is fine and the unigram distribution. Now, can we do maybe something better than just a unigram distribution there? And this is the idea of the Kneser-Ney smoothing. So, let us see this example. This is the malt or this is the Kong. So, the word Kong might be even more popular than the word malt but the thing is, that it can only occur in a bigram Hong Kong. So, the word Kong is not very variative in terms of different contexts that can go before it. And this why, we should not prefer this word here to continue our phrase. On the opposite, The word malt is not that popular but it can go nicely with different contexts. So, this idea is formalized with the formula in the top of this slide. Let us have the probability of the words proportional to how many different contexts can go just before the word. So, if you take your absolute discounting model and instead of unigram distribution have these nice distribution you will get Kneser-Ney smoothing. Awesome. We have just covered several smoothing techniques from simple, like, Add-one smoothing to really advanced techniques like, Kneser-Ney smoothing. Actually, Kneser-Ney smoothing is a really strong baseline in language modeling. So, in the next lessons we will also cover Neural language models and we will see that it is not so easy to beat this baseline</td>
    <td></td>
  </tr>
</table>


* [Perplexity computation](https://www.coursera.org/learn/language-processing/supplement/fdxeI/perplexity-computation)

<table>
  <tr>
    <td>This is an optional reading about perplexity computation to make sure you remember the material of the videos and you are in a good shape for the quiz of this week.
Perplexity is a popular quality measure of language models. We can calculate it using the formula:
\mathcal{P} = p(w_{test})^{- \frac{1}{N} }, \ \text{where} \ p(w_{test}) = \prod\limits_{i=1}^{N+1} p(w_i|w_{i-n+1}^{i-1})P=p(wtest​)−N1​, where p(wtest​)=i=1∏N+1​p(wi​∣wi−n+1i−1​)
Recall that all words of the corpus are concatenated and indexed in the range from 1 to NN. So NN here is the length of the test corpus. Also recall that the tokens out of the range are fake start/end tokens to make the model correct.
Check yourself: how many start and end tokens do we have in a trigram model?
Now, if just one probability in the formula above is equal to zero, the whole probability of the test corpus is zero as well, so the perplexity is infinite. To avoid this problem, we can use different methods of smoothing. One of them is Laplacian smoothing (add-1 smoothing), which estimates probabilities with the formula:
p(w_i|w^{i-1}_{i-n+1}) = \frac{c(w^i_{i-n+1}) + 1}{c(w^{i-1}_{i-n+1}) + V}p(wi​∣wi−n+1i−1​)=c(wi−n+1i−1​)+Vc(wi−n+1i​)+1​
Note, that VV here is the number of possible continuations of the sequence w_{i-n+1}^{i-1}wi−n+1i−1​, so VV is the number of unique unigrams in the train corpus plus 1. Do you see why? Well, we include the fake end token to this number, because the model tries to predict it each time, just us as any other word. And we do not include the start tokens, because they serve only as a prefix for the first probabilities.
Now, let’s review the following task together.
Task:
Apply add-one smoothing to the trigram language model trained on the sentence:
"This is the cat that killed the rat that ate the malt that lay in the house that Jack built."
Find the perplexity of this smoothed model on the test sentence:
“This is the house that Jack built.”
Solution:
We have n=3, so we will add two start tokens <s1>, <s2> and one end token <end>.
Note, that we add (n-1) start tokens, since the start tokens are needed to condition the probability of the first word on them. The role of the end token is different and we always add just one end token. It's needed to be able to finish the sentence in the generative process at some point.
So, what we have is:
train: <s1> <s2> This is the cat that killed the rat that ate the malt that lay in the house that Jack built <end>
test: <s1> <s2> This is the house that Jack built <end>
Number of unique unigrams in train is 14, so V = 14 + 1 = 15.
Number of words in the test sentence is 7, so N = 7.
\mathcal{P} = p(w_{test})^{- \frac{1}{N} }, \ \text{where}\ p(w_{test}) = \prod\limits_{i=1}^{8} p(w_i|w_{i-2} w_{i-1}) = \prod\limits_{i=1}^{8} \frac{c(w_{i-2} w_{i-1} w_i) + 1}{c(w_{i-2} w_{i-1}) + 15}P=p(wtest​)−N1​, where p(wtest​)=i=1∏8​p(wi​∣wi−2​wi−1​)=i=1∏8​c(wi−2​wi−1​)+15c(wi−2​wi−1​wi​)+1​
All right, now we need to compute 8 conditional probabilities. We can do it straightforwardly or notice a few things to make our life easier.
First, note that all bigrams from the test sentence occur in the train sentence exactly once, which means we have (1 + 15) in all denominators.
Also note, that "is the house" is the only trigram from the test sentence that is not present in the train sentence. The corresponding probability is p(house | is the) = (0 + 1) / (1 + 15) = 0.0625.
All other trigrams from the test sentence occur in the train sentence exactly once. So their conditional probabilities will be equal to (1 + 1) / (1 + 15) = 0.125.
In this way, perplexity is (0.0625 * 0.125 ^ 7) ^ (-1/7) = 11.89.
The quiz of this week might seem to involve heavy computations, but actually it does not, if you think a bit more :) Good luck!</td>
    <td></td>
  </tr>
</table>


    4. Sequence tagging with probabilistic models

        * [Hidden Markov models](https://www.coursera.org/learn/language-processing/lecture/cNdwa/hidden-markov-models)

<table>
  <tr>
    <td>Hey, everyone. In this lesson, we are going to discuss sequence tagging task. We'll start with some examples and then investigate one model that can be used to solve it. It will be called Hidden Markov Model. Let us get started. So the problem is as follows. You are given a sequence of tokens, and you would like to generate a sequence of labels for them. The examples would be part of speech tagging, named entity recognition, or semantic slot filling that we have briefly seen in introduction. Now, for example, I am given a sequence of words, and they want to produce part-of-speech tags like here. For example, saw is that the verb and I is the pronoun and so on. There are different systems to list all possible tags and different systems to decide what are important tags and not Important tags. One system is here, and you can see that there are some open class words and closed class words and some other labels. For example, we have also here some punctuation and symbols for punctuation if we see it in the texts. Another example would be named entity recognition. So, here, I have a sequence and last Friday or Winnie-the-Pooh would be some named entities. Sometimes we really need to find them in the texts and use them as features, or maybe we need them to generate answer to some question. What kind of named entities can we see? First of all, it would be some persons or organization names or locations but not only them. For example, we can also have dates and the units and any other entities that you see in the texts. What kind of approaches work well for these type of tasks? First, it would be rule-based approach, and we do not cover it in details here. And the second one would be just to take classifiers like naive Bayes classifier or maybe a logistic regression and use them separately at each position to classify the labels for this position. This is obviously not super nice because you do not use the information about sequence, right? So a better idea would be to do sequence modeling, and this is what is our lesson about. And another idea would be to do neural networks, and you will hear about them in another lesson. To come to our models, to hidden Markov models, let us define everything properly. So please remember this notation, and let us go through it. So we have a sequence. It will be a denoted by x, and we have a sequence of tags, which would be denoted by y. The length of the sequence is big T, and by T, we will denote the positions in our sequence. Now, the task is to produce y given x, to produce the most probable y. So we need to find the most probable tags for our words in our model, but to find something most probable, we should first define the model and see what are the probabilities here. So we are going to define the joint probability of x and y. Do you understand the equation in the bottom of the slide? So take a moment to see that, actually, the right-hand side and the left-hand side can be both used to find the argmax just because they are different just by probability of x, which does not depend on y. So if I multiply the left side by probability of x, it is just constant for me. So I don't care whether I do it or not. Let us define the model. This is probably the most important slide of our video because it tells us what is the hidden Markov model. So we need to produce the probabilities of x and y. x are our observable variables and y are our hidden variables. Now, first, we apply product rule to decompose it into two parts. Then every part should be farther simplified, and we can do this by applying two different assumptions. The first assumption, which is called Markov assumption, will help us to model the probability over the sequence of tags. So we say that this sequence can be factorized into probabilities of separate pieces so we just model the probabilities of the next tag given the previous tag. You have already seen this in the y-gram language models. So there, we applied exactly the same formula to model the probabilities of words. Now, we do this to model the probabilities of tags. Also there, maybe you remember that we had to have some special start token so that we could write down this like that and be with the fact that we can have T equal to zero. So, here, the first term would be the probability of the first tag given some special zero tag, which is just the start tag. Awesome. Now, the second assumption is about output probabilities. So we need to produce the probabilities of x given y and it is rather complicated. But we can say that we will factorize it into the probabilities for separate positions. And for every position, we will have the probability of the current word given the current tag. So given these two assumptions, you can write down the formula in the top of the slide. And this is the definition of hidden Markov model. Please take a moment to remember it. Now, hidden Markov model can be used to model texts. So it can be used to generate texts. Let us see how it happens. So, first, we'll need to generate a sequence of tags, and after that, we will generate some words given current tags. For example, we will start with our fake start tag, and then we'll generate some tags using those transition probabilities. So we generate them and continue like that, and after we are done, we start to generate words given this tags. So we generate a sequence, and we see that the model has generated some nice example from Winnie The Pooh here. Now, let us define once again what are the parameters of the model. So hidden Markov model is defined by the following formal five steps. So, first, we have some hidden states, those y in our previous notation. We have some finite set of this states and also we have a special start state as zero. We have transition probabilities that model the probabilities of the next state given the previous one in our model. Then we have some vocabulary of output words, and for them, we have output probabilities, so the probabilities of words given tags. Now, can it compute how many parameters do you have? Well, actually, we have just two matrices of parameters. A metrics is N plus one by N because you have also this special start token. And for this special start state, you need to model the probabilities to transit to all other states. For the output probabilities, the dimension will be the number of states multiplied by the number of output words. So we have lots of parameters. We need to learn them somehow. So how can we train the model? Well, let us imagine for now that we have a supervised training data set, which means that we have a sequence of words, and we have a sequence of tags for this words. Then we could just count, so we would count how many times we see the tag Si, which is followed by the tag Sj. And we will normalize this count by the number of times when we see the tag as i. So this way, we will get the conditional estimate to see the tag as j after the tag as i. We could do the similar thing with output probabilities. So there, we would count how many times some particular tag generates some particular output and normalize this by the number of times where we see this particular tag. So this way, we would get conditional probabilities for the output words. So I can tell you that this intuitive estimates are also maximum likelihood estimates, but let us get in a little bit more details and make sure that we understand how we compute these counts. So we have many sentences. Let us just concatenate all those sentences into one long corpus of the length big T. Then we can compute those counts just by running through the corpus from T equal to one to big T and computing the indicator functions. So there, in green, I write down that I see the tag Si on the position Yt minus one and then I see the tag as j that follows and the similar indicator function in the denominator. So this is still the same formula, just another Notation. But the thing is that in real life, usually, you do not have tags in your training data. So the only thing that you'll see is plain text, and you still need to train hidden Markov model somehow. How can you do it? Well, obviously, you cannot estimate those indicator functions because you don't see the tags of the positions, but you could try to approximate those indicators by some probabilities. So compare this formula with the formula in the bottom of this slide. The only thing that has changed is that instead of indicators, I have probabilities now. So something in between zero and one, but how can we get these probabilities? Well, if they have some trained hidden Markov model, we could try to apply it to our texts to produce those probabilities. So the E-step in the top of this slide says that given some trained hidden Markov model, we can produce the probability to see tags Si and Sj in the position T. So this is something like three-dimensional array. T, i, and j would be the indices, and it can be actually done effectively with dynamic programming. The only thing is that we need to have trained model there. So the clever idea is to alternate two steps. The E-step says that let us fix some current parameters of the Model A and B matrices and use them to generate those estimates for probabilities of tags for every certain position. And M-step says let us fix those estimates and use them to update our parameters, and those parameters updates are actually still maximum likelihood estimates in this case. So in the slide, I have everything written down for A matrix, but you can obviously do very similar things to compute B matrix. So this is just a sketch of the algorithm it is called Baum-Welch algorithm. And this is just a case over more general algorithm, which is called EM algorithm, and we will see this algorithm several times during our course every time when we have some hidden variables that we do not see in the training data and some observable variables. This is all about training the hidden Markov model. I hope you have got some flavor of how it can be done. And then in the next video, we will discuss how to apply the hidden Markov model once it's trained.</td>
    <td></td>
  </tr>
</table>


        * [Viterbi algorithm: what are the most probable tags?](https://www.coursera.org/learn/language-processing/lecture/FMAba/viterbi-algorithm-what-are-the-most-probable-tags)

<table>
  <tr>
    <td>[MUSIC] Hey, in the previous video we discussed
that there is Hidden Markov Model that can be used for part-of-speech tagging. We have briefly discussed
how to train this model, now how can we apply this to our texts? This slide is just a motivation for the next video, to show you that it
is actually not that simple problem. So for example, you can see a sequence
of words, a bear likes honey, and you can see that it can be
decoded in different ways. So the sequence of text in the top and in
the bottom of the slides are both valid. So both sequences could generate
this piece of text, right? This is because we have
very ambiguous language. For example, likes can be noun, or verb, or something else maybe,
if we think about it. Okay, then how can we generate the most probable sequence of text
given a piece of text? This is called decoding problem, and this
is formalized in the bottom of the slide. So we need to find y which maximizes
the probability of y given x. And as we have briefly discussed,
it would be the same as y that maximizes the probability
of both variables. Now could we probably just
compute the probabilities of all ys, and then choose the best one? Well not actually, because this brute force approach
would be really slow, right? Or maybe not feasible at all, because you have very many
different sequences of text. Let's say you have big T which is
the length of the sentence, and you have 10 possible states. Then you will have T to the power
of 10 different sequences, right? So this is too much, but
fortunately there is an algorithm based on dynamic programming
that can help us to do this effectively. Before we go to it, I just want to recap
the main formula for Hidden Markov Model. So please remember that we have the
probabilities of y given the previous y, which are called transition probabilities,
and then we have output probabilities. And we multiply them by
the positions to get our total probability of both variables. Okay, so
this is probably the main slide here, which tells you what is Viterbi algorithm. Let us go through it slowly. We would like to find
a sequence of text that is the most probable to generate
our output sentence. However, we can do it straightforwardly. So let us first start with finding
the most probable sequences of text, up to the moment t,
that finishes in the state s. So those big Qt,s things. And let us denote by small qt,s
the probabilities of those sequences. Now, how can we compute those
probabilities effectively? We can say that once we have computed
them for the time moment t-1, we can use that to compute the next
probabilities for the next time moment. How do we do this? Well we see that we have our
transition probabilities and output probabilities,
as is said by the Hidden Markov Model. And then we have these green qt-1
probabilities that hide the transition probabilities and output probabilities for
all the previous time moments. Okay, and
then we apply maximum because remember, we are only interested in
the most probable sequences. Now when we compute these
probabilities we are also interested in the argmax,
because the argmax will tell us what are the states where these
probabilities can be found. Okay, probably it is not yet clear for
you, so during the video I am going to show you lots of pictures and examples
that explain how this algorithm works. Let us say that we have some Hidden Markov
Model, and it is already trained for us. So this is A matrix, which has probabilities of some states
given some other states, right? Our transition probabilities. Each row sums into 1, which means that
we have indeed correct probabilities. Now I have a question for you, do you think that something
is missing in this matrix? So if you remember from
the previous video, we had also our special start state. And we will need to have a special
first row in this matrix that will have the probabilities of some states
given this special start state. For now, let's say that this row is
just filled with equal probabilities, so it is just a uniform distribution. Okay, so we have these parameters for
A matrix, and we have this B matrix which
tells us the probabilities of the output tokens given some states. Awesome, now let us imagine
that our probability algorithm has already computed
the probabilities up to some moment. So we have some probabilities
up to the bear in our sequence. Now how can we compute the probabilities
for the next time moment? For the first state, for ADJ state, we can try to go from
different previous states. And we have transition probabilities,
different transition probabilities for each of them. And we have also the output probabilities
to generate likes in this case. Now we need to find maximum, so
we need to find the best way. And in this case, it will be this one. So we find this way, and we say that the probability is
now composed by three things. The previous probability that we had so
far, the transition probability, and the output probability. Now let us try to do the same for the next state. So another state is NOUN, and we again need to compare different paths. So we have three paths, which one would be chosen this time? So this time we will choose this one, because the multiplication of three components again will be maximum for this one, right? So this will be that value, and we could compute this. Now we perform the same thing for the last state. So here we have this path, we choose this one, and compute the probabilities. And are we done? Well we could compute the probabilities for the next time moment, and this way we can move forward. But it is very important to remember the best transitions that we used. Why? Because this is how we are going to find our best path for the states. This is what we compute for every time moment. Now let us zoom out and see what is happening for the whole sentence. So this is what you get if you compute all the probabilities and remember all the best transitions. Now once again, what are those best transitions? So for example, the path that goes through ADJ, NOUN, VERB and NOUN is the most probable sequence of text that generates our sentence and finishes in the state NOUN, okay? This is just the definition. And the probability of this path is written down there, near the NOUN. Okay, so what do we need to compute now? We have these three candidates that can finish in one of the three states for honey. Remember, we need the most probable one ever. So we need to compute maximum once again. We need to compute maximum, it would be NOUN. And then we take it and we say that okay, the last state in our sequence should be NOUN. And then we backtrace to VERB, to NOUN, and to ADJ to get the sequence of text that is the best in this case. Awesome, we are done with the pictures. This slide just summarizes everything that we have already said. So to create your best path you need to first allocate an array queue of the dimension the number of words in your sentence, by the number of states in your model. Then you need to fill the first column in your matrix, right? So for the first time moment it is rather easy to compute the probabilities of the states. This would be just the probabilities to come to these states from the initial start state, multiplied by the probability to output the first word in this current state. Now after that you have a loop. So you go through all your positions in time, and you go through all your steps, and compute those max values and argmax values. After that you have your last column. You apply argmax once again to find what would be the pick for the last word in your sentence. After that you backtrace all your text for the path. And you are done, you get your best path. So this is decoding in Hidden Markov Models. And this is real useful, not only in part-of-speech tagging actually, but also, for example, for some signals and other source of data where you have different states that can generate some outputs. So this is all for this video. And in the next one, we will discuss a few more models that are similar to Hidden Markov Model</td>
    <td></td>
  </tr>
</table>


        * [MEMMs, CRFs and other sequential models for Named Entity Recognition](https://www.coursera.org/learn/language-processing/lecture/Ctjm2/memms-crfs-and-other-sequential-models-for-named-entity-recognition)

<table>
  <tr>
    <td>In this video, we will cover several probabilistic graphical models for sequence tagging tasks. You already know one of them. So you know Hidden Markov models, and you briefly know how to train and apply this. In this video, we'll speak about few more and we'll apply them to Named Entity Recognition, which is a good example of sequence tagging tasks. So, this is a recap for hidden Markov model. You maybe remember the formula, and one important thing to tell you is that it is generative model, which means that it models the joint probabilities of x and y. And the picture in the bottom of the slide illustrates the formula. So you see that every arrow is about some particular probability in the formula. So we have transition probabilities going from one y to the next one, and we have output probabilities that go to x variables. Now, another one would be Maximum Entropy Markov model. Is a super similar. So, do you see what is now different in the picture? Only one thing has changed. You have now the arrows going from x to ys. Okay? And in the formula, we can see that now the factorization is a little bit different. We have the probabilities of current tag given the previous tag, and the current x. What is also important is that this model is discriminative, which means that it models the conditional probability of y given x. So, it doesn't care to describe how the text can be generated. It says that the text is observable and we just need to produce the probabilities for our hidden variables, y. Now, another important thing to mention is that you see that it is factorized nicely still. Right? So you have some separate probabilities, and you have a product of this probabilities. Let us look into more details how every probability can be written down. So, in this model, every probability looks like that. Maybe a little bit scaring but let us see what is here. So, we have some exponents and we have some normalization constants. So, this is actually just soft marks. This is just soft marks applied to some brackets. What do we see in the brackets? We have there something linear. We have weights multiplied by features. So you can think about a vector of weights and the vector of features, and you have a dot product. Probably you have a feeling that you have already seen something similar just in the machine learning. So, do you remember a similar model? Well, actually logistic regression is super similar to maximum entropy Markov model. There, you also have a soft max applied to dot product of features and weights. The only difference is that here, you have rather complicated features that can depend on the next and the previous states. So, the model knows about the sequence, and it knows that the tags are not just separate. Actually this feature is a very interesting question because it is our job to generate these features somehow, so we will get back to this question in a few slides. But now, let us write down one more probabilistic graphical model that can be even more powerful than that one. This model is called conditional random field. First, you can see that it is still discriminative model. So it is the probability of y given x. Now you can see that it is actually similar to the previous one, for example, in the brackets, you'll still have these dot product of weights and features. But what is the difference? Do you see an important difference between CRF and maximum entropy Markov model? The thing is that now you have only one normalization constrain that goes outside of the product. So you don't have any probabilities inside at all. So, the model is not factorized into probabilities. Instead, we have some product of some energy functions, and then we normalize all of them to get the final probability. And this normalization is actually complicated because, well, we have many different sequences, and we have to normalize in such a way that these probability sums to one over all possible sequences of tags. Now, when we depict this model with the graph, it would be undirected graph. So, I don't have any arrows at all. I have just some links between the nodes. And actually in this picture, I write down a more specific case than the one in the top of the slide. So here, in the top of the slide, you can see that your features can depend on three things. And here I kept them only for two things. So I have one type of features about transitions and another type of features about outputs. Obviously, I could have something more complicated. So, a general form of conditional random field would look like that. So you have some arbitrary factors that depend on some groups of y variables and x variables. In the picture, these small green boxes would stand for different factors, that you multiply them and then you normalize them and you get your probabilities. So this is rather general model, and maybe you are already lost with all those options. So, if this is the case, I have good news for you. Probably, you will not need to implement this model yourself because there are lots of black-box implementations for CRF model. So, this is just some links to check out. For example, Mallet is a nice library that has an implementation for CRF, but what we have to do is we have to generate features to feed them into CRF models. So, in the rest of the video, we'll discuss how to generate those features. From the formulas, you might remember that those "f" features can depend on three things; the current tag, the previous tag, and the current output. Now, not to be overwhelmed with the variety of your features, there is a very nice common technique which is called label observation features. So, it says that you are only allowed to have these kind of features. The observation part is about something that depends on the output. So, we will go to this part, and the green part, the labeled part, is about indicators. So you just check whether you have the current label equal to y, and you check it for all possible labels. Okay? So, it means that you have as many features as many labels you have multiplied by the number of different observation functions that you invent. And in the case of the second and the third line, you will have even more features because there, you check these indicators for the current and for the previous tags. So, we are going to have lots of them. Now, how those observation parts will look like. This is just some example taken from the paper, and it says that you can be as creative as you want. So, first, you can check that your current word is equal to some predefined word. And you can check it for all the words in the vocabulary. So again, you will have let's say plus 100,000 features just by the first line. Then, you may want to check your part-of-speech tag for the current word defined by some extrinsic part-of-speech tager, and you will have again many features, many binary features here that tell you whether your tag is equal to noun or whether it is equal to a verb and so on and so on for all possible tags. And you can have lots of other ways to define your features. For example, you can check whether your word is capitalized, or whether it has a dash in the middle, or whether it consists only from capital letters, or whether it appears in some predefined list of stop words or predefined list of some names. So, actually this is how your work would look like if you decided to use CRF for some sequence tagging task. You would take some implementation and you would generate as many useful features as you can think about. Now, one trick that I want to mention here is that even though we say that the feature can depend only on the current output, x_t, well, it is not like that. So, honestly, we can put into this x_t everything that we want. For example, we can say that our current x_t consists of the current word, the previous word and the next word, and with have features for all of them. So, you should multiply those tremendous number of features by three right now. And it is okay. So, the model will not break down just because it is discriminative model. So, we do not care about modeling the sequence of x which means that we can do whatever we want basically. So, this picture depicts this idea and it says that every feature can actually have access to all words in our sentence. It can be a little bit messy. So, just be prepared that it happens and people think that this is okay and this is nice, and this is how it works actually. So, just to sum up what we have discussed. Well, we have discussed different graphical models for sequence tagging. We have discussed one of them, Hidden Markov model with more details. So, we have seen how to train this model and how to do decoding in this model. And in practice, usually you will need just to do feature generation, and use some implementation of one of this model. So, this is all for probabilistic approach for sequence tagging, but we will also discuss neural networks for this task later.</td>
    <td></td>
  </tr>
</table>


    5. Deep Learning for the same tasks

        * [Neural Language Models](https://www.coursera.org/learn/language-processing/lecture/bAJan/neural-language-models)

<table>
  <tr>
    <td>Hi! During this week, you have already learnt about traditional NLP methods for such tasks as a language modeling or part of speech tagging or named-entity recognition. So in this lesson, we are going to cover the same tasks but with neural networks. So neural networks is a very strong technique, and they give state of the art performance now for these kind of tasks. So please stay with me for this lesson. This is just the recap of what we have for language modeling. So the task is to predict next words, given some previous words, and we know that, for example, with 4-gram language model, we can do this just by counting the n-grams and normalizing them. Now, let us take a closer look and let us discuss a very important problem here. Imagine that you have some data, and you have some similar words in this data like good and great here. In our current model, we treat these words just as separate items. So for us, they are just separate indices in the vocabulary or let us say this in terms of neural language models. You have one-hot encoding, which means that you encode your words with a long, long vector of the vocabulary size, and you have zeros in this vector and just one non-zero element, which corresponds to the index of the words. So this encoding is not very nice. Why? Imagine that you see "have a good day" a lot of times in your data, but you have never seen "have a great day". So if you could understand that good and great are similar, you could probably estimate some very good probabilities for "have a great day" even though you have never seen this. Just by saying okay, maybe "have a great day" behaves exactly the same way as "have a good day" because they're similar, but if it reads the words independently, you cannot do this. I want you to realize that it is really a huge problem because the language is really variative. Just another example, let us say we have lots of breeds of dogs, you can never assume that you have all this breeds of dogs in your data, but maybe you have dog in your data. So if you just know that they are somehow similar, you can know how some particular types of dogs occur in data just by transferring your knowledge from dogs. Great. What can we do about it? Well, this is called distributed representations, and this is exactly about fixing this problem. So now, we are going to represent our words with their low-dimensional vectors. So that dimension will be m, something like 300 or maybe 1000 at most, and this vectors will be dense. And we are going to learn this vectors. Importantly, we will hope that similar words will have similar vectors. For example, good and great will be similar, and dog will be not similar to them. I ask you to remember this notation in the bottom of the slide, so the C matrix will be built by this vector representations, and each row will correspond to some words. So we are going to define probabilistic model of data using these distributed representations. And we are going to learn lots of parameters including these distributed representations. This is the model that tries to do this. Actually, this is a very famous model from 2003 by Bengio, and this model is one of the first neural probabilistic language models. So this slide maybe not very understandable for yo. That's okay. I just want you to get the idea of the big picture. So you have your words in the bottom, and you feed them to your neural network. So first, you encode them with the C matrix, then some computations occur, and after that, you have a long y vector in the top of the slide. So this vector has as many elements as words in the vocabulary, and every element correspond to the probability of these certain words in your model. Now, let us go in more details, and let us see what are the formulas for the bottom, the middle, and the top part of this neural network. Looks scary, isn't it? Don't be scared. I will break it down for you. So the last thing that we do in our neural network is softmax. We apply to the components of y vector. The y vector is as long as the size of the vocabulary, which means that we will get some probabilities normalized over words in the vocabulary, and that's what we need. What happens in the middle of our neural network? There is some huge computations here with lots of parameters. Actually, every letter in this line is some parameters, either matrix or vector. The only letter which is not parameters is x,. So what is x? X is the representation of our context. You remember our C matrix, which is just distributed representation of words. So you take the representations of all the words in your context, and you concatenate them, and you get x. So just once again from bottom to the top this time. You get your context representation. You feed it to your neural network to compute y and you normalize it to get probabilities. Now, to check that we understand everything, it's always very good to try to understand the dimensions of all the matrices here. For example,what is the dimension of W matrix? Well, we can write it down like that, and we can see that what we want to get in the result of this formula, has the dimension of the size of the vocabulary. Now what is the dimension of x? Well, x is the concatenation of m dimensional representations of n minus 1 words from the context. So it is m multiplied by n minus 1. Here you go. You can see the dimension of W matrix. So this neural network is great, but it is kind of over-complicated. So you can see that you have some non-linearities here, and it can be really time-consuming to compute this. So the next slide is about a model which is simpler. Let's try to understand this one. It is called log-bilinear language model. Maybe it doesn't look like something more simpler but it is. So let us figure out what happens here. You still have some softmax, so you still produce some probabilities, but you have some other values to normalize. So you have some bias term b, which is not important now. The important part is the multiplication of word representation and context representation. Let's figure out what are they. So the word representation is easy. It's just the row of your C matrix. What is the context representation? You still get your rows of the C matrix to represent individual words in the context, but then you multiply them by Wk matrices, and this matrices are different for different positions in the context. So it's actually a nice model. It is not a bag-of-words model. It tries to capture somehow that words that just go before your target words can influence the probability in some other way than those words that are somewhere far away in the history. So you get your word representation and context representation. And then you just have dot product of them to compute the similarity, and you normalize this similarity. So the model is very intuitive. It predicts those words that are similar to the context. Great. This is all for feedforward neural networks for language modeling. The next video is about recurrent neural networks. So see you there.</td>
    <td></td>
  </tr>
</table>


        * [Whether you need to predict a next word or a label - LSTM is here to help](https://www.coursera.org/learn/language-processing/lecture/tRWMp/whether-you-need-to-predict-a-next-word-or-a-label-lstm-is-here-to-help)

<table>
  <tr>
    <td>[MUSIC] Hi, this video is about
a super powerful technique, which is called recurrent neural networks. I assume that you have heard about it,
but just to be on the same page. So this is a technique that
helps you to model sequences. You have an input sequence of x and
you have an output sequence of y. Importantly, you have also
some hidden states which is h. So here you can know how you transit
from one hidden layer to the next one. So this is just some activation
function f applied to a linear combination of the previous
hidden state and the current input. Now, how do you output
something from your network? Well, this is just a linear layer
applied to your hidden state. So, you just multiply your
hidden layer by U metrics, which transforms your hidden
state to your output y vector. Great, how can we apply this network for
language bundling? Well, actually straightforwardly. So the input is just some
part of our sequence and we need to output the next
part of this sequence. What is the dimension of those U
metrics from the previous slide? Well, we need to get the probabilities
of different watts in our vocabulary. So the dimension will be
the size of hidden layer by the size our output vocabulary. Okay, so we apply softmax and
we get our probabilities. Okay, how do we train this model? So in the picture you can see that
actually we know the target word, this is day, and this is wi for
us in the formulas. What does the model, the model outputs
the probabilities of any word for this position? So, we need somehow to compare our work, probability distribution and
our target distribution. So, the target distribution
is just one for day and zeros for
all the other words in the vocabulary. And we compare this to distributions
by cross-entropy loss. You can see that we have a sum there
over all words in the vocabulary, but this sum is actually a fake sum because
you have only one non-zero term there. And this non-zero term corresponds
to the day, to the target word, and you have the probable logarithm for
the probability of this word there. Okay, so the cross-center is probably one
of the most commonly used losses ever for classification. So maybe you have seen it for
the case of two classes. Usually there you have just
labels like zero and ones, and you have the label multiplied by
some logarithm plus one minus label multiplied by some other logarithms. Here, this is just the general case for
many classes. Okay, so, we get some understanding
how we can train our model. Now, how can we generate text? How can we use our model
once it's trained? We need some ideas here. So the idea is that,
let's start with just fake talking, with end of sentence talking. And let's try to predict some words. So we get our probability distribution. How do we get one word out of it? Well we can take argmax. This is the easiest way. So let's stick to it for now. Now what can we do next? Next, and this is important. We can feed this output words as
an input for the next state like that. And we can produce the next
word by our network. So we continue like this
we produce next and next words, and
we get some output sequence. Now we took argmax every time. So it was kind of a greedy approach, why? Because when you will see your sequence,
have a good day, you generated it. Well probably it's not the sequence
with the highest probability. Because you could, maybe at some step,
take some other word, but then you would get a reward during
the next step because you would get a high probability for some other output
given your previous words. So something that can be better than
greedy search here is called beam search. So beam search doesn't try to estimate the
probabilities of all possible sequences, because it's just not possible,
they are too many of them. But beam search tries to keep
in mind several sequences, so at every step you'll have, for example five
base sequences with highest possibilities. And you try to continue
them in different ways. You continue them in different ways,
you compare the probabilities, and you stick to five best sequences,
after this moment again. And you go on like this,
always keeping five best sequences and you can result in a sequence which is
better than just greedy argmax approach. Okay, so what's next? Next I want to show you the experiment
that was held and this is the experiment that compares recurrent network model with
Knesser-Ney smoothing language model. So you remember Knesser-Ney
smoothing from our first videos. And here this is 5-gram language model. You can see that when we add
recurrent neural network here we get improvement in perplexity and
in word error rate. So this is nice. This says that recurrent neural networks
can be very helpful for language modeling. And one interesting thing is that,
actually we can apply them, not only to word level, but
even to characters level. So instead of producing the probability of
the next word, giving five previous words, we would produce the probability
of the next character, given five previous characters. And this is how this model works. So this is the Shakespeare corpus that you have already seen. And you can see that this character-level recurrent neural network can remember some structure of the text. So you have some turns, multiple turns in the dialog, and this is awesome I think. Okay, so this is just vanilla recurring neural network, but in practice, maybe you want to do something more. You want some other tips and tricks to make your awesome language model work. So first thing to remember is that probably you want to use long short term memory networks and use gradient clipping. Why is it important? Well you might know about the problem of exploding gradients or gradients. And this architectures can help you to deal with this problems. If you do not remember LSTM model, you can check out this blog post which is a great explanation of LSTM. You can start with just one layer LSTM, but maybe then you want to stack several layers like three or four layers. And maybe you need some residual connections that allow you to skip the layers. Now another important thing to keep in mind is regularization. You could hear about drop out. For example, in our first course in the specialization, the paper provided here is about dropout applied for recurrent neural networks. Well, if you don't want to think about it a lot, you can just check out the tutorial. Which actually implements exactly this model and it will be something working for you just straight away. And maybe the only thing that you want to do is to tune optimization procedure there. So you can use rate and decent, you can use different learning rates there, or you can play with other optimizers like Adam, for example. And given this, you will have a really nice working language model. And most likely it will be enough for your any application. However, if you want to do some research, you should be aware of papers that appear every month. So this is has just two very recent papers about some some tricks for LSTMs to achieve even better performance. So this is kind of really cutting edge networks there. So this is a lot of links to explore for you, feel free to check it out, and for this video I'm going just to show you one more example how to use LSTM. This example will be about sequence tagging task. So you have heard about part of speech tagging and named entity recognition. And this is one more task which is called symmetrical labelling. Imagine you have some sequence like, book a table for three in Domino's pizza. Now, you want to find some symantic slots like book a table is an action, and three is a number of persons, and Domino's pizza is the location. Usually use B-I-O notation here which says that we have some beginning of the slowed sound inside the slot and just outside talkings that do not belong to any slot at all, like for and in here. I want to show you that my directional is LSTM as super helpful for this task. So, what is a bi-directional LSTM? Well you can imagine just LSTM that goes from left to the right, and then another LSTM that goes from right to the left. Then you stack them, so you just concatenate the layers, the hidden layers, and you get your layer of the bi-directional LSTM. After that, you can apply one or more linear layers on top and get your predictions. And you train this model with cross-entropy as usual. So nothing magical. Okay, what is important here is that this model gives you an opportunity to get your sequence of text. It can be this semantic role labels or named entity text or any other text which you can imagine. And one thing I want you to understand after our course is how to use some methods for certain tasks. Or to see what are the state of other things for certain tasks. So for this sequences taking tasks, you can use either bi-directional LSTMs or conditional random fields. So these are kind of two main approaches. Conditionally random fields are definitely older approach, so it is not so popular in the papers right now. But actually there are some hybrid approaches, like you get your bidirectional LSTM to generate features, and then you feed it to CRF, to conditional random field to get the output. So if you come across this task in your real life, maybe you just want to go and implement bi-directional LSTM. And this is all for this week. Thank you</td>
    <td></td>
  </tr>
</table>


        * [Recognize named entities on Twitter with LSTMs](https://www.coursera.org/learn/language-processing/peer/982Gp/recognize-named-entities-on-twitter-with-lstms)

<table>
  <tr>
    <td>The assignment and all necessary instructions are provided in the IPython notebook on our github. Please, make sure that you have the latest version of the assignment by downloading the ipython notebook (+all supplementary scripts) right before you start working on it!
Frequently Asked Questionsless 
Here are several frequently asked questions about the assignment and review process. Read these before starting your assignment.
In this assignment you will use TensorFlow library. If you are not familiar with that, we think that you might try some manual, for example, this one. We tried to adapt our assignment for all learners and you will find all necessary links to useful functions in documentation.
It is not necessary to strictly follow all recommended hyperparameters, but we cannot guarantee that otherwise you will have comparable or better results.
Our network will be trained on small data and the time of training is about 5-10 minutes</td>
    <td></td>
  </tr>
</table>


3. [Week 3](https://www.coursera.org/learn/language-processing/home/week/3)

    6. Word and sentence embeddings

        * [Distributional semantics: bee and honey vs. bee an bumblebee](https://www.coursera.org/learn/language-processing/lecture/PtRav/distributional-semantics-bee-and-honey-vs-bee-an-bumblebee)

<table>
  <tr>
    <td>[MUSIC] Hey everyone, you're very welcome
to week three of our course. This week is about semantics,
so we are going to understand how to get the meaning of words, or
documents, or some other pieces of text. We are going to represent
this meaning by some vectors, in such a way that similar words
will have similar vectors, and similar documents will
have similar vectors. Why do we need this? Well for example, we need this in search. So let's say we want to do some ranking. For example, we have some keywords, and
then we have some candidates to rank. And then we can just compute these kind
of similarities between our query and our candidates, and
then get the top most similar results. And actually there are numerous of
applications of these techniques. For example, you can also think
about some ontology learning. What it means is that sometimes you need
to represent the hierarchical structure of some area. You need to know that there
are some concepts, and there are some examples of these concepts. For example, you might want to know that,
I don't know, there are plumbers, and that they can fix tap or faucet. And you need to know that tap and faucet are similar words that
present the same concept. This can be also done by
distributional semantics, and this is what we are going
to cover right now. Okay, so for example we want to understand
that bee and bumblebee are similar. How can we get that? Let us start with counting
some word co-occurrences. So we can just decide
that we are interested in the words that co-occur
in a small sliding window. For example, in a window of size ten. And if the words co-occur
we say plus 1 for this counter, and
get these green counters in the slide. So this way we will understand
that bee and honey are related. They are called syntagmatic associates because they often co-occur
together in some contexts. However if we get back to our
example to understand that tap and faucet are similar,
that's not what we need. We need just to get to know some other
second order co-occurrence, which means that these two words would co-occur
with similar words in their contexts. For example, we can compute a long,
sparse vector for bee, the cells, what are the most
popular neighbors of this word? And we will also count the same vector for
bumblebee. And after that, we will compute
similarity between these two vectors. This way we will understand that bee and bumblebee can be interchangeably
used in the language. And this means that they are similar,
right? So they're usually called
paradigmatic parallels, and this is the type of co-occurrence
that we usually need. Now let us get in
a little bit more details first on how to compute
those green counts. Okay, so as I have already said,
you can compute just word co-occurrences. But they can be biased because of
too popular words in the vocabulary, like stop words, and then it will
be rather noisy estimates, right? So you need some help to
penalize too popular words. And then one way to do this would
be Pointwise Mutual Information. It says that you should put the individual
counts of the words to the denominator. This way you will understand
whether these two words are randomly co-occurrent or not. So if you look to the first formula, you see that in the numerator you have
the joint probability of the words. And in the denominator you
have the joint probability in the case that the two random
variables are independent, right? So if the words were independent, then we could just say that this is
two probabilities, it is factorized. So in case of independent words,
you will get 1 there for this fraction. And in case of dependent words
that occur too much together, you will get something more. So this is the intuition of PMI. This is whether the words are randomly
co-occurred or they are really related. Now do you see any more
problems with this measure? Well actually, there are some. So when you see some counts and
logarithm applied to them, you should have some bad feeling that
you are going to have 0 somewhere. And indeed you can have those words
that never co-occur together, or those words that co-occur really rare,
and then very low numbers for your logarithms. So the good idea is just to say,
let us take the maximum of the PMI and 0. This way we will get rid of
those minus infinity values, and we will get nice positive
Pointwise Mutual Information. This is the measure that is usually used,
and the idea that goes for all these measures actually would
be just distributional hypothesis. It says that you can know
the word by the company it keeps. So the meaning of the word is somehow
defined by the context of this word. Now let's get back to this nice slide so
we know how to compute those green values. What other problems we have here? Well if you want to measure a cosine
similarity between these long, sparse vectors,
maybe it's not a good idea. So it is long, it is noisy,
it is too sparse. Let us try to do some
dimensionality reduction. So here, the 
matrix on the left is just stacked rows that you have seen during the previous slide. It is filled with some values, as we have described like PMI, and then we factorize it into two matrices. And the dimension there in between of them would be K, so it is some low dimensional factorization. For example, K would be 300, or something like that. You have lots of different options how to do this factorization, and we will get into them later. What we need to know now is that we are going to compare now the rows of v matrix instead of the original sparse rows of X matrix. This way we will get some measure of whether the words are similar, and this will be the output of our model. So far we have looked into how our words occur with other words from a sliding window. So we had some contexts, which would be words from a sliding window. However, we can have some more complicated notion of the contexts. For example, you can have some syntax parses. Then you will know that you have some syntactic dependencies between the words. And you can see that some word has co-occurred with another word and had this type of relationship between them, right? So for example, Australian has co-occurred with scientist as a modifier. So in this model, you will say that your contexts are word plus the type of the relationship. And in this case you will have not a square matrix, but some matrix of words by contexts. And for the contexts you will have the vocabulary of those word class modifier units, okay? This will be actually a better idea to do, because syntax can really help you to understand what is important to local context and what is not. What is just some random co-occurrences that are near, but that are not meaningful. However, usually we just forget about it and we speak about word by word co-occurrence matrix. But still, we will sometimes say that we have words and contexts, because in the general model we could have that.</td>
    <td></td>
  </tr>
</table>


        * [Explicit and implicit matrix factorization](https://www.coursera.org/learn/language-processing/lecture/A4eQD/explicit-and-implicit-matrix-factorization)

<table>
  <tr>
    <td>[MUSIC] Hey, in the previous video,
you could see that to build models of distributional semantics we need
some kind of matrix factorization. And now we are going to cover different
ways to perform matrix factorization in many details. Let us start with an approach based
on singular value decomposition. So this is just a fact from linear
algebra that says that every matrix can be factorized into
those three matrices. The left and the right matrices
will contain so called left and right eigenvectors. And the metrics in the middle
will be diagonal and the values on the diagonal will
be related to eigenvalues. Now importantly, those values on the diagonal will
be sorted in the decreasing order. And how many of them do we have there? Well, the number of those
values on the diagonal will be the number of non-zero
eigenvalues of X transposed X. So it is related to
the rank of that matrix. Awesome, so this is just something that we
know, but how can we use this in our task? The idea is as follows. So once those values are sorted,
what if we keep first k components? Probably, if we keep just this
blue regions of every matrix, we will get some nice approximation
of the original X matrix. Now, it sounds rather hand-wavey, right? What is a nice approximation? Actually, there is a very
accurate answer to this question. So always blue part is the best
approximation of rank k for X matrics, in terms of the loss that
is written in the bottom of this slide. Let us look into this loss,
this is just squared loss, okay? So you can see that there are some
squares between corresponding elements of two matrices, and
we can then pair the squared root. So I hope it sounds familiar to you. And now we'll take away ease
that truncated SVD can get for us the best approximation of the matrix,
according to this loss. Now do you remember that, actually,
in the previous video, we were going to factorize our matrix into two matrices,
not into three matrices, okay. Well, what can we do about it? We can just use some heuristics, actually. So one idea would be to say,
okay, let us take the first and the second matrix and
put them to phi matrix. And then the last one and
put it to theta, or another option would be to say that
the diagonal matrix in between should be honestly split between
two matrices phi and theta. So we apply squared root and say that
one squared root goes to the left and another squared root goes to the right. Okay, so you can see that SVD provides
some ways to build phi and theta matrices. Now let us just summarize again
what we have just realized. We were going to build models
of distributional semantics, which means we have some word concurrence
matrix filled with PMI values or with some concurrence values. And we were going to represent it with
a factorization of phi and theta matrices. And then we could see that SVD can
provide this for us, and you can actually see that those phi u vectors and
theta v vector would be our embeddings. Will be our what vectors that
we are interested in, and also if you just multiply phi u and
theta v, as a inner product,
you will get the value equal to, let's say,
PMI in the left-hand side matrix, right? So I just want you to
remember this notation. We have phi and theta matrices. We have phi u and theta v vectors. And the matrix way of thinking
about it corresponds to the way of thinking about every
element like PMI is somehow equal to the dot product of phi u and
theta v. Okay, awesome. So far we have been using the squared loss
because SVD deals with squared loss but this is not perfect maybe,
maybe we can do better. So the next model, which is called
global vectors, tries to do better. So don't be scared. Let us break it down. You can see that it is still some
squared loss, but it is weighted. So f function provides some weights and
you'll see that f function is increasing. So if the word concurrences are higher,
then this particular element of the matrix is more
important to be approximated exactly. Okay, now I have a question for you. Why does this green line for f function stops increasing at some
point and just goes as a constant? Well, you might remember about
stop words that are not important. So starting from some moment,
the words should not get bigger weights here just
because this is somehow noisy. Awesome, now let us look
into the brackets and see that we have a green part and
the red part there. So the red part is our regional matrix. We used to have there word concurrences or
BMI values, now we have logarithms. Okay, why not? Now what is the green part? So usually we would have just the inner
product of phi u and theta v. This would correspond to our
matrix factorization task. Now it's almost the same. We just have those b terms
that are some bias terms. This is not actually important but we say, well, maybe we should chew on them as well
and have some more freedom in our model. But this is again not so important. Now, how do we train this model? In case of SVD, we have just an exact recipe from linear algebra how to build those three matrices. Now, we have just some loss that we want to minimize. What do we do with some losses that you want to minimize? Well, we can try to do stochastic gradient descent. In this case, we can treat every element of the matrix as some example. So we take an element, we perform one step of gradient descent and then update the parameters and take another element, and proceed in this way. Finally, we will get our global vectors model trained and we'll obtain those phi u and theta v vectors vectors that can be used as word embeddings. And actually, this model is rather recent and very nice way to provide word embeddings, so please keep it in mind. Now, let us forget about matrix factorization for a few slides. Let us think about what would be other approaches to build word embeddings. Another approach to think about the task would be language modeling. So just a recap, language modelling is about probabilities of some words given some other words. So in this case, in case of so called skip-gram model, we are going to produce the probability of context given some focus word. For example, the context words can come from some window of a fixed size. Now, we assumed that this window is represented as a bag of words. So that's why we'll just go to this product and have probabilities of one word given another word. Now how would we produce these probabilities. You can see the formula in the bottom of the slide, and you might recognize that it is softmax. Okay, so it means that it will be indeed normalized as we want, as a correct probability distribution over the vocabulary. Now what is their insight? Again, inner products between phi u and theta v. So these inner products correspond to some similarity between two vectors. We take this similarity, normalize them using softmax, and get probabilities. Okay, so this model is called Skip-gram, it is very important, but unfortunately, it is rather slow. Because softmax computation is slow especially if you have a big vocabulary. What can we do about it? Actually we can just say let us reformulate the whole procedure like that. You can see that we have some green and red parts of this slide. The green part corresponds to positive examples. The positive examples are about those pairs of words, u and v, that co-occurred together. We just take it from data, right? And for this pair of words that co-occur, we want to predict yes and there are some other pairs of words that do not co-occur, and we just sample them randomly. And for them, we want to predict no. Now how do we model these predictions? We model them with sigmoid function. So you'll see that now, sigmoid function is applied to inner products. It will give us probabilities whether yes or no. Okay, now maybe you are somehow scared with the mathematical expectation that you see here. And actually, we do not take any mathematical expectation though we write it in some theoretical way, but what we do we just sample. So k will correspond to the number of samples, and we sample just k words from the vocabulary for every given u word. And we use all those samples in this term, so you can just forget about expectation for now. What do we get by this model? Well, again we build those embeddings for words, and now we do not need to normalize anything by the size of the vocabulary, and this is nice. So this model, skip-gram negative sampling, is very efficient and used a lot. The final slide for you is somehow to understand that this skip-gram negative sampling model is still related to matrix factorization approach that we have discussed. What if we take the derivatives of this loss? So we say that, what would be the best value for the inner product? If you do this, you will understand that the best value for this inner product is a shifted PMI value, like here. So you'll see that it is PMI value minus logarithm of k, where k is the number of negative samples. This is just some nice fact. This was published in a recent paper and it says that even though in skip-gram negative sampling model, we do not think about any matrices. However, you can still interpret it as some implicit matrix factorization of shifted PMI values into our usual two matrices, phi and theta. So this is rather important fact because it says that more recent models are still very similar to more previous models, like SVD decomposition of PMI values. Because now, the only thing that is different is that we have shifted PMI and some other loss. Okay, this all for the mathematical part of the methods, and in the next video, we will see how to use these models
</td>
    <td></td>
  </tr>
</table>


        * [Word2vec and doc2vec (and how to evaluate them)](https://www.coursera.org/learn/language-processing/lecture/F9M3C/word2vec-and-doc2vec-and-how-to-evaluate-them)

<table>
  <tr>
    <td>[MUSIC] Hey, in the previous video, we had all necessary background to see
what is inside word2vec and doc2vec. These two models are rather famous, so
we will see how to use them in some tasks. Okay, let us get started with word2vec. So it is just some software package
that has several different variance. One variant would be
continuous bag-of-words. It means that we try to predict
one word given the context words. Another option would be to do
vice versa and predict context words given some words and
this one will be called skip-gram. Then softmax computation
is usually too slow, and producing those
probabilities is not effective. So there are some ways to avoid that,
and one way would be negative sampling. So you might remember
from the previous video, that we have already discussed
skip-gram negative sampling model. And this is one of architectures
of this word2vec program. This is open source, so
you can just find the code there. Okay, now how do we use these models? One task would be to produce some
meaningful similarities between words. So you remember that we
could build word embeddings, sum vectors that represent
the meaning of the word. Now if we just apply cosine
similarity to those vectors, we will get some measure of
similarity between the words. How can we test this model? How can we see that actually those
similarity measures are good and somehow meaningful? Well this is actually a very
complicated question, but we can use some human judgements. So what we see is that
there are some data sets provided by linguists that look
like the first table in this slide. For example, they say that tiger and
tiger are super similar. And media and radio are also similar,
but not to that extent, and so on. So you have some ranked list of
word pairs with their similarities. Now you can produce the more similarities
by your model as the table in the right. And then just compare these two rank list,
let's say with Spearman's correlation. And then you will see whether your model
somehow agrees with the assessors. Obviously, using human judgements
is not always the best way. It would be better to use
some extrinsic evaluation. For example,
you could build a ranking system and then apply word similarities there, compute the quality of the ranking system,
and use this to evaluate your model. Okay, anyways,
let us come to the next task. The next task is rather appealing. So if you have not heard about it, look,
we have some vectors for the words. For example, we have a vector for king, then we can apply some arithmetic
operations over these vectors. Like king minus man plus woman,
we get some other vector, and the closest word for
this vector will be queen, you see? So we can somehow understand
relations between the words. We can understand that man to woman is
related in the same way as king to queen. You can think about some
other analogy like, for example, Moscow minus Russia plus
France will be equal to Peru. And something like that, when I say equal
it means that those cosine similarity. Gets its maximum for the target worth. This task become very famous
after the recent papers. However it have been started
a lot in cognition science and it was called relational similarity. On the contrary, the similarity
that we have been discussing up to this moment was called
attributional similarity. Now how do you evaluate
word analogies task? Again, we usually rely on human judgments. So there are some datasets that
say that man to woman relates the same as king to queen and
so on, for many, many examples. And then we try to
predict the last word and compute the accuracy of these predictions. Awesome, now let us see how different
models perform on these two tasks. So let us try to remember
what is every model about. So the first row is about PMI, we can
compute PMI values between the words and just the long sparse vectors
of PMI as word embeddings. Second, we can apply SVD
to the PMI matrix and get somehow, dense and
low dimensional vectors. Then we can do skip-gram
negative sampling module, that we have discussed a lot
in the previous video. Now, do remember what is GLoVe? Well, GloVe was also covered
in the previous video. And it was about measures factorization
with respect to weighted squared loss. So you might remember this green F
function that was increasing and at some point, it just went constant not
to be overwhelmed with too frequent words. Okay, so we have four methods,
different ways to perform matrix factorization maybe implicit matrix
factorization and obtain word embeddings. And you can see that actually
they perform really similar. So different columns here correspond
to different datasets of word pairs. So you can see that the bold best values
are somehow spread around this table. So very old methods like SVD
is not much worse than very recent methods like
skip-gram negative sampling. Now what havens with word analogies task? There are also two data sets here,
one from Google and another from Microsoft Research. And one take away of it would be that the quality is very nice. So for Google that is said is about 70% of accuray, which means that in 70%, we can guess the right word correctly. For example, we can guess that king minus man plus woman is equal to queen. This is awesome, but actually we'll see some problems with that in the next video. Okay, now let us come to paragraph2vec or doc2vec. Actually these two names are about the same model. Paragraph2vec name goes from the paper. Doc2vec name goes from gensim library where it is implemented. You remember that in word2vec, we had two architectures to say that we produce contexts given some focus word or vice versa focus word given some contexts. Now we can also have some document ID. So we will treat the document the same way as we treated words. So we will have some ID in some fixed vocabulary of the documents and then we will build embeddings for the documents. Now there are again, two architecture. The first architecture, DM, stands for providing the probabilities of focus words, given everything we have. And DBOW architecture stands for providing the probability of the context given the document. So the last one is somehow similar to skip-gram model, right? But instead of the focus words, we condition on the document. Now, how can we use this model? Well we can use it to provide some documents similarities and apply, for example, for ranking again. How can it test that we document similarities provided by our model a good? Well we need some test set again, so that it does set released by the way paper in the bottom of the slide provides triplets from archive papers. We have some paper, and then another paper, which is known to be similar, and then a third paper which is dissimilar. So the task is to predict if this one is the dissimilar one. And if the model can do this, then the model provides good estimates for document similarities. And so we'll just compute the accuracy of this prediction task, Okay? Now I want just to sum up everything that we have covered. So there are models called word2vec and doc2vec, that actually not even models but rather implementations of different architectures. You can find them, for example, in gensim library and play with them. And there are different ways to use this model. And for every usage of the model, we need some dataset to evaluate whether the usage will be good. Whether the provided word similarities or document similarities will be good enough. Some other ways to evaluate these models would be to see whether each component of the vector is interpretable in some way or to look into the geometry of this space of the embeddings. This might be more complicated, so we are not going into the details of these ways, and maybe it is also not so needed. So one takeaway that is really needed to be understood is that count-based methods like SVD applied to PMI metrics are not so different from predictive based methods as word2vec. So there is no magic behind them, and in the next video, we will actually see some problems behind them. Thank you</td>
    <td></td>
  </tr>
</table>


        * [Word analogies without magic: king – man + woman != queen](https://www.coursera.org/learn/language-processing/lecture/lpSIA/word-analogies-without-magic-king-man-woman-queen)

<table>
  <tr>
    <td>Hey, in the previous video,
you have seen that what to school and works nicely for lots of different tasks. However, in this video we will raise some
doubts and we will see that especially for world analogies task everything is not so
smooth. Just to recap, the word2vec model is
trained in an unsupervised manner. It means that the model just
sees the let's say Wikipedia. And it this to obtain word vectors. Now the word vectors that are obtained
have some nice properties. For example, if you take the vector for
king, you'll subtract the vector for man and add the vector for
woman, you will get the vector. And the closest word t
this one will be queen. And this is awesome, right? So it looks like the model could
understand some meaning of the language. Even though we did not have
this in the data explicitly. But, well let us look
into more closer details. How this of the closest word is performed? So you see that we have these
arithmetic expression and then we maximize cosine similarity
between the result of the expression and all the candidates in our space but we
exclude three candidates from this search. So we say that our source was, so let's say king, man and
woman do not participate in this search. And well, you know what is this rather
important trick that is usually omitted in descriptions of word2vec model. However, let us see what would
happen if we looked into this honestly and if we performed
the maximization in the whole space. The picture shows what would
be the closest neighbor to the arithmetic expression
in case of the search. The color shows the ratio. The names on the left correspond to
different categories of analogies. It is not so important for now, let us look into the last
one which is called Encyclopedia. The example about king
will fall into this one. So what we see is that when we do king
minus man plus woman we get some vector and in most of cases it will be close
to the b vector, which is king here. Also in some cases, it can be close to
a prime vector, which is woman here. But never to b prime vector,
which is our target queen vector. So you see actually in the, let's say
90% or 80% of different analogies. We find the vector which is
close to b vector instead of the target b prime vector. Well you know it somehow ruins
a little bot the picture that word2vec understands our language. Now I want to dig a little
bit deeper into it. How can it be that when we exclude a,
a prime and b vectors,
we actually find b prime vectors. But if we do not exclude them,
we end up with b vector, so I think that this picture
can shed some light. The thing is that the shift vector
a prime minus a seems to be close to 0. So this plus a woman
minus men is close to 0. It means that when we employ our and we try to find the closest neighbor,
well the closest neighbor is actually b. But once b is excluded, the next
closest neighbor is indeed b prime. And we say that, okay,
king is excluded and queen is found. Okay, so maybe we can just use much
more simple methods to do this. I mean,
can we just the nearest neighbor of b? And do not apply any
arithmetic operations at all. Well, some people tried that and they said that for
one particular category of analogies. The plural category which is apples to
apples is same as orange to oranges. Just the strategy to take the closest
neighbor of b results in 70% accuracy. So you see this is a really high accuracy,
very similar to what we could see for world2vec back in the previous videos. And just by a very dumb approach. This is another visualization all for
the same idea. So this comes from a recent paper and
it says let us split our word analogy examples
into several brackets. So for example,
those analogy examples where b and b prime vectors are similar will be
going to the right and those examples, where b and b prime vectors are also
similar, we will be going to the left. Now the blue bars in this slide show
the accuracy of wold2vec for every bucket. So you can easily see that the blue
bars are high on the right, and low on the left. Which means that word2vec
works really nice in those analogies where b and b prime are similar. And it works poorly for those more complicated tasks
where they are not similar. Now let us see what are those
more complicated tasks? So let us study what types of
analogies covered in this diagram. There are actually four
main types of analogies. For example you can find actor and
actress in the very bottom line. This is kind of the same thing
as our king and queen example. But we have much more here. So first,
we have some morphological examples. We have inflectional morphology which
means that we can just change the form of the word like orange to oranges
is the same as apple to apples. Or we can have derivational morphology,
which can also change the part of speech, like bake to
baker is the same as play to player. Now we have lots of different
semantical analogies. For example, we have hypernyms there. This would be, for example,
peach to fruit is the same as cucumber to vegetable. We have many more, for example, the nice one is about colors like blood is red and sky is blue. And there have many different options, and this is not so easy to build this dataset, so we need some linguistic expertise. Anyways, once we have this, can we look into how word2vec performs for different analogies. And can we compare word2vec with a very simple baseline. The baseline would be to just to take the closest neighbor to one of the query words. So here we go. Each line here corresponds to some analogy example. For example, one line could correspond to apple to apples is the same as orange to oranges. Now the left point for every line is the performance of the baseline. And the right point of every line is the performance of word2vec. So it means that horizontal lines show you that word2vec is not better than base line. When the line has a high slope, it means that word2vec does a good job. So you see that for inflectional morphology which is an easier task. What the and for derivational morphology. All the lines are horizontal. Now what happens with semantic analogies. Well this is a nice picture, so the thing on the left is about different types of analogies and most of them have horizontal slow push means that word2vec doesn't work for them. But two lines, red lines have high slope and those two are the examples about genders. Like man to woman is as king to queen is as actor to actress and so on. And the picture on the right is about some named entities and the three red lines are about countries and capitals. Examples that are really popular in world2vec descriptions. For example, Moscow to Russia is the same as Paris to France. So you know what? Those very famous examples are kind of the only ones that actually work with Word2vec. I mean there are not the only ones but it looks things are generally worst in random for different tasks. Okay, so the takeaway of this insight would be that you should be very careful about hype that you see around. So it is always nice to dig into some details, like how is a relation performed? What would happen with a little bit different tasks? And see whether some of these provide some good or bad solutions. So to me it looks like word2vec works nicely for word similarity task. For example, if you have some application where you need to understand that tap and faucet are really similar and should be placed into one category then word2vec is your choice. But you shouldn't be blinded, and you shouldn't think that it somehow solves the language or provides the solutions for word analogy task in all the cases. So it works sometimes, but not always. And this is a nice question to have further research on it. Okay, in the next video, we'll talk about some extensions of those techniques like word2vec. We will see what are now current state of that approaches and what are some open source implementations that you can use in your applications. So stay tuned, and we will get some practical advice, what models to build in your cases. [SOUND]</td>
    <td></td>
  </tr>
</table>


        * [Why words? From character to sentence embeddings](https://www.coursera.org/learn/language-processing/lecture/yqddn/why-words-from-character-to-sentence-embeddings)

<table>
  <tr>
    <td>Hey. We have just covered lots of ways to build word embeddings. But, you know what? Why words? I mean, sometimes we need representations for sentences, and it's not so obvious so far how to get that, and in some cases, we need to go to sub-word level. For example, we might have a language with rich morphology and then it would be nice to somehow use this morphology in our models. Actually, linguistics can be really helpful. So, we will see a couple of examples right here. Let us start with morphology. So, for example, in English you can say mature or immature, and then relevant, irrelevant and so on. So, you know that there are some prefixes that can change the meaning of the word to the opposite one. So we will have antonyms. Now, on the other hand, you can understand that there are some suffixes that do not change the semantics of the words a lot. For example, I have no break and the breaker are still about the similar concepts in some sense. So the idea of the window proposed in the paper in the bottom of the slide is the following. Let us try to put the words that have some not important morphological changes together in the space, and on the opposite, let us try to have the embeddings for words that have some prefixes that change the meaning of the word completely. Let us put them as far as possible. So you see you try to put some words closer and some words more far away. You can do this in many ways like, let's say it would be some regularization of your model, or you will have some loss, and then at some other losses, to make sure that you have this additional constraints. Okay? This idea is nice, but sometimes we don't have linguists to tell us what is the morphological patterns in the language. What can we do then? Well, in this case, we can try just to have more brute-force approach. This would be to go to character n-grams. This is FastText model proposed by Facebook research, and this is really famous just because it has a good implementation and you can play with it. So, the idea is as follows. Let us represent a word by a set of character n-grams, and also let us put the word itself to this set as well. For example, for character n-grams of size three, we'll have this example in this slide. Usually, we will have several n-values, like n from three to six. And we will have n-grams of different length in the same set, and this set will represent the word. Now, how can we use this? Well, if you remember in skip-gram negative sampling, we had some similarity between two words and we could represent it just by those product of these words. Now, what if the words are represented not by vectors but by set of vectors? Well, we can sum. So, we can say that now we have a sum over all those character n-grams, and every character n-gram is represented by the vector. Awesome. So, I think the idea is great, and it works well for languages with rich morphology. So, FastText model provides a nice way to represent sub-word unions. Now, what if we need to go to another level and to represent sentences? Do you have any ideas how to build sentence embeddings? There are some ideas summarized in this slide. So, the more simple ideas would be, what if we just take the pre-trained vectors, let's say, from Word2Vec model and average them to obtain the embedding of the sentence? Well, you might have also some weighted average, for example with TF-IDF weights. But, you know what? It might be not too nice approach because those pre-trained vectors are trained with some other objectives, and they might not suit well for our task. So, another idea would be somehow to represent the sentences as a sum of sub-sentence units. Let's have a closer look. First, we are going to represent the similarity between word and the sentence, and our training data will be those words that occur in some sentences. So it will be a positive pair, word occurred in a sentence. The negative example will be some word that occurs in some other sentence. So we assume that they are not similar. Now, how do we model this similarity? Again, we have a sum over sub-unions. So, these unions will be word n-grams. And, a bag of word n-grams will represent a sentence. Awesome. So, you see that this model is very similar to FastText model, but instead of having character n-grams to represent words, you have word n-grams to represent sentences. Also, another minor difference is that now you have average not the sum. We have this one divided by the size of the set, but this is not so important. So you see that in different levels of our language we can have some similar ideas. What if we build some general approach for all these levels? An attempt to build this general approach is found in a very recent paper which is called StarSpace. So, the idea here is that we have some entities represented by features. For example, words represented by character n-grams or sentences represented by word n-grams. But you can go further. For example, we can think about recommendation systems. There we have users, and they are represented as bag of items that they like. For example, as bag of movies. So we'll learn how to embed users and movies in the same space. Another example, it would be document classification problem. So there you have documents as a bag of words and you have labels, for example, sentiment labels, and these are rather simple entities that are represented by a singleton feature, the label itself. So, in this application, you will try to learn to produce correct labels for document. So, you'll say that the similarity measure between the documents and the label should be high if this label can be found in the supervised data for this document, and low vice-versa. So you build the model and you get the embeddings for labels, and documents, and words in the same space. Now, you can read about all those applications on the GitHub page. But I want to cover in more details just one application. And this will be, again, sentence embeddings. So let's say, we have some supervised data about similar sentences. So we know that some group of sentences are duplicates and they are similar. Let us put them into one line of the file. Let us have tabs between the sentences and let us have spaces between the words. Now, in this format, we can feed these data to StarSpace and say that we need to train the embeddings for words and sentences. Then, what happens next? Well, the similar sentences are the good source for positive examples. And we will just take two sentences from the line and use them as a positive example in our model. Now these similar sentences can be just sampled at random, for example we take a sentence from one line and just a random sentence from another line and say that they are a negative example. Then we train those kind of word2vec models in some sense, and obtain the embeddings for all our entities. Awesome. So, the last thing that I want to cover is Deep learning approaches to build sentence representations. So, actually, everything up to this point, we are rather shallow networks. So if we speak about deep learning, we could have three main trends here. One trend would be, obviously, recurrent neural networks that are popular in NLP. Another would be convolutional neural networks that are actually much faster than recurrent neural networks so it seems like it is a super promising approach. And the third one would be recursive neural networks or so-called Tree-LSTMs or Dynamic Acyclic Graph, DAG-LSTM. So, these kind of models use the syntax of the language to build some hierarchical representations. These are rather awesome approaches. We will not have too much time to cover them, but you just need to know that syntax can help us to build the representation of the sentence. Now, the take-away of this slide would be that linguistics can help us, for example as morphology syntax in many many tasks. The last architecture that I want to cover in this video is called skip-thought vectors. And it is based on recurrent neural networks. So the idea is as follows. You have some sentence, and you want to predict the next sentence. You encode your sentence with a recurrent neural network and get some hidden representation. It is called thought vector. Now, once you have these, you try to generate the next sentence with the language model. So you already know that there are neural language models. Now it is a conditional neural language model conditioned on this thought vector. And the great thing is that this thought vector is going to represent the meaning of the sentence, and it can be used as the embedding. Actually, this architecture is called encoder-decoder architecture. And we will have many many details about it in the next week. So, if you haven't realized all the details just from one slide, don't worry, we will cover them in many many details.</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/bK2iE/word-and-sentence-embeddings)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


        * [Programming Assignment: Find duplicate questions on StackOverflow by their embeddings](https://www.coursera.org/learn/language-processing/programming/3wbz5/find-duplicate-questions-on-stackoverflow-by-their-embeddings)

<table>
  <tr>
    <td>Find code on github</td>
    <td></td>
  </tr>
</table>


    7. Topic models

        * [Topic modeling: a way to navigate through text collections](https://www.coursera.org/learn/language-processing/lecture/fzTUI/topic-modeling-a-way-to-navigate-through-text-collections)

<table>
  <tr>
    <td>Hi everyone. This week, we have explored a lot of ways to build vector representations for words or for some pieces of text. This lesson is about topic modeling. Topic modeling is an alternative way to build vector representations for your document collections. So, let us start with a brief introduction to the task. You are given a text collection and you want to build some hidden representation. So, you want to say that okay there are some topics here and every document is described with those topics that are discussed in this document. Now, what is the topic? Well you can imagine that you can describe a topic with some words. For example, such topic as weather is described with sky, rain, sun, and something like this and such topics such as mathematics is described with some mathematical terms and probably they do not even overlap at all. So, you can think about it as soft biclustering. Why soft b-clustering? So, first, it is biclustering because you cluster both words and documents. Second, it is soft because you will see that we are going to build some probability distributions to softly assign words and documents to topics. This is the formal way of saying the same thing. You are given a text collection, so you are given the counts of how many times every word occurs in every document? And what you need to find is two kinds of probability distributions. So, first the probability distribution over words for topics and second the probability distribution over topics for documents. And importantly, this is just the definition of a topic, so you should not think that topic is something complicated like it is in like real life or as linguists can say. For us, for all this lesson, topic is just a probability distribution. That's it. Where do we need this kind of models in real life? Well actually everywhere because everywhere you have big collections of documents. It can be some news flows or some social media messages or maybe some data for your domain like for example papers, research papers that you do not want to read but you want to know that there are some papers about these and that and they are connected this way. So, you want some nice overview of the area to build this automatically and topic models can do it for you. Some other applications would be social network analysis or even dialog systems because you can imagine that you would generate some text. You know how to generate text, right, from the previous week, but now you can do this text generation dependent on the topics that you want to mention. So, there are many many other applications, for example aggregation of news flows, when you have some news about politics for example and you want to say that this topic becomes popular nowadays, and one other important application that i want to mention is exploratory search, which means that you want to say this is some document that I am interested in, could you please find some similar documents and tell me how they are interconnected? Now, let us do some math, so let us discuss probabilistic latent semantic analysis, PLSA. This is a topic model proposed by Thomas Hofmann in 1999. This is a very basic model that tries to predict words in documents and it does so by a mixture of topics. So, do you understand what happens for the first equation here in this formula? Well, this is a law of total probability. So, if you just don't care about documents in the formulas for now, about D, you can notice that this is the law of total probability applied here. Just take a moment to understand this. Now what about the second equation here? Well, this is not correct, this is just our assumption. So, just for simplicity, we assume that the probability of word given the topic doesn't depend anymore on the document. So, this is conditional independence assumption. This is all that we need to introduce PLSA model. Now i just want you to give you intuition on how that works. So, this is a generative story. This is a story how the data is generated by our model. You have some probability distribution of topics for the document and first you decide what would be the topic for the next word. Then, once you have decided on that, you can draw a certain word from the probability distribution for this topic. So, this model just assumes that the text is generated not by authors, not just by handwriting, but by some probability procedure. So, first we toss a coin and decide what topic will be next, and then we toss a coin again and decide what would be the exact word, and we go on through the whole text. Well, this is just one way to think about it. If you do not feel very comfortable with this way, I will provide for you another way of thinking. So, this is a matrix way of thinking about this same model. You can imagine that you have some data which is just word document co-occurrences. So, you know how many times each word occurs in each document. That's why you can compute distributions. You can compute probabilities of words in documents. You just normalize those counts and that's it. Now you need to factorize this real matrix into two matrices of your parameters, Phi and Theta. One matrix, Phi matrix, is about probability distributions over words and Theta matrix contains probability distributions over topics. Actually every column in this matrix is a probability distribution. So, this is just a matrix form of the same formula in the top of the slide, and you can see that it holds just for one element and for any element obviously. So, this is the introduction of the model and in the next video we will figure out how to train this model. So stay with me.</td>
    <td></td>
  </tr>
</table>


        * [How to train PLSA?](https://www.coursera.org/learn/language-processing/lecture/OzmrF/how-to-train-plsa)

<table>
  <tr>
    <td>Hey. Let us understand how to train PLSA model. So, just to recap, this is a topic model that predicts words in documents by a mixture of topics. So we have some parameters in this model. We have two kinds of probability distributions, phi parameters stand for probabilities of words and topics, and theta parameters stand for probabilities of topics and documents. Now, you have your probabilistic model of data, and you have your data. How do you train your models? So, how do you estimate the parameters? Likelihood maximization is something that always help us. So the top line for this slide is the log-likelihood of our model, and we need to maximize this with the respect to our parameters. Now, let us do some modification in this formula. So first, let us apply logarithm, and we will have the sum of logarithms instead of the logarithm of the products. Then, let us just get rid of the probability of the document because the probability of the document does not depend on our parameters, which they do not even know how to model this pairs. So we just forget about them. What we care about is the probabilities of words in documents. So we substitute them by the sum of our topics. So this is what our model says. Great. So that's it. And we want to maximize this likelihood, and we need to remember about constraints. So our parameters are probabilities. That's why they need to be non-negative, and they need to be a normalized. Now, you can notice that this term that we need to maximize is not very nice. We have a logarithms for the sum, and this is something ugly that is not really clear how to maximize. But fortunately, we have EM-algorithm, you could hear about this algorithm in other course in our Specialization. But now, I want just to come to this algorithm intuitively. So let us start with some data. So we are going to train our model on plain text. So this is everything of what we have. Now, let us remember that we know the generative model. So we assume that every word in this text has some one topic that was generated when we decided to reach what will be next. So let us pretend, just for a moment, just for one slide, that we know these topics. So let us pretend that we know that the words sky, raining, and clear up go from sub topic number 22, and that's it. So we know these assignments. How would you then calculate the probabilities of words in topics? So you know you have four words for this topic, and you want to calculate the probability of sky, let's say. This is how you do it. You just say, "Well, I like one word out of these four words. So the probability will be one divided by four." By NWT here, I denote the count of how many times this certain word was connected to this certain topic. So, can you imagine how would we evaluate the probability of topics in this document for this colorful case. Well, it's just the same. So we know that we have four words about this red topic, and we have 54 words in our document, that's why we have this probability for this example. Well, unfortunately, life is not like this. We do not know this colorful topic assignments. What we have is just plain text. And that's a problem. But, can we somehow estimate those assignments? Can we somehow estimate the probabilities of the colors for every word? Yes we can. So, Bayes rule helps us here. What we can do, we can say that we need probabilities of topics for each word in each document and apply Bayes rule and product rule. So, to understand this, I just advise you to forget about D in all this formulas, and then everything will be very clear. So we just apply these two rules, and we get some estimates for probabilities of our hidden variables, probabilities of topics. Now, it's time to put everything together. So, we have EM-algorithm which has two steps, E-step and M-step. Each step is about estimating the probabilities of hidden variables, and this is what we have just discussed. M-step is about those updates for parameters. So we have discussed it for the simple case when we know the topics assignment exactly. Now, we do not know them exactly. So, it is a bit more complicated to compute NWT counts. This is not just how many times the word is connected with this topic, but it's still doable. So, we just take the words, we take the counts of the words, and we weight them with the probabilities that we know from the E-step. And that's how we get some estimates for NWT. So this is not int counter anymore. It has some flow to variable that still has the same meaning, still has the same intuition. So, the EM-algorithm is a super powerful technique, and it can be used any time when you have your model, you have your observable data, and you have some hidden variables. So, this is all formulas that we need for now. You just want to understand that to build your topic model, you need to repeat those E-step and M-step iteratively. So, you scan your data, you compute probabilities of topics using your current parameters, then you update parameters using your current probabilities of topics and you repeat this again and again. And this iterative process converge and hopefully, you will get your nice topic model trained.</td>
    <td></td>
  </tr>
</table>


        * [The zoo of topic models](https://www.coursera.org/learn/language-processing/lecture/vraEh/the-zoo-of-topic-models)

<table>
  <tr>
    <td>Hey. You know the basic topic model which is called PLSA, and now you know how to train it. Now, what are some other topic models in this world? What are some other applications that we can solve with the topic modeling? I want to start with a nice application. It is about diary of Martha Ballard. So, this is a big diary. She was writing for 27 years. This is why it's rather complicated for people to read this diary and to analyze this. So, some other people decided to apply topic modeling to this and see what other topics revealed in this diary. These are some examples of the topics and you can see just the top most probable words. So, you remember you have your Phi metrics which stand for probabilities of words and topics. And this is exactly those words with the highest probabilities. And actually you can see that the topics are rather intuitively interpretable. So, there is something about the gardens, and potatoes, and work in these gardens. There is something about shopping like sugar, or flour, or something else. So, you can look through these top words, and you can name the topics, and that's nice. What's nicer, you can look into how these topics change over time. So, for example the gardening topic is very popular during summer, in her diary, and it's not very popular during winter, and it makes perfect sense. Right? Another topic which is about emotions has some high probabilities during those periods of her life when she had some emotional events. For example, one moment of high probability there corresponds to the moment when she got her husband into prison, and somebody else died, and something else happened. So, the historians can I say that, ''OK, this is interpretable. We understand why this topic has high probability there.'' Now, to feel flexible and to apply your topics in many applications, we need to do a little bit more math. So, first, this is the model called Latent Dirichlet Allocation, and I guess this is the most popular topic model ever. So, it was proposed in 2003 by David Blei, and actually any paper about topic models now cite this work. But, you know this is not very different from PLSA model. So, everything that it says is that, ''OK we will still have Phi and Theta parameters, but we are going to have Dirichlet priors for them.'' So, Dirichlet distribution has rather ugly form and you do not need to memorize this, you can just always Google it. But, important thing here is that we say that our parameters are not just fixed values, they have some distribution. That's why as the output of our model, we are also going to have some distribution over parameters. So, not just two matrices of values, but distribution over them, and this will be called posterior distribution and it will be also Dirichlet but with some other hyperparameters. In other course of our specialization devoted to Bayesian methods, you could learn about lots of ways how to estimate this model and how to train it. So, here I just name a few ways. One way would be a Variational Bayes. Another way would be Gibbs Sampling. All of them have lots of complicated math, so we are not going to these details right now. Instead, I'm going just to show you what is the main path for developing new topic models. So, usually people use probabilistic graphical models and Bayesian inference to provide new topic models and they say, ''OK, we will have more parameters, we will have more priors. They will be connected to this and that way.'' So people draw this nice pictures about what happens in the models. And again, let us not go into the math details but instead let us look how these models can be applied. Well, one extension of LDA model would be Hierarchical topic model. So, you can imagine that you want your topics to build some hierarchy. For example, the topic about speech recognition would be a subtopic for the topic about algorithms. And you see that the root topic has some very general Lexis and this is actually not surprising. So, unfortunately, general Lexis is always something that we see with high probabilities, especially for root topics. And in some models, you can try to distill your topics and to say well maybe we should have some separate topics about the stop words and we don't want to see them in our main topics, so we can also play with it. Now, another important extension of topic models is Dynamic topic models. So, these are models that say that topics can evolve over time. So, you have some keywords for the topic in one year and they change for the other year. Or you can see how the probability of the topics changes. For example, you have some news flow and you know that some topic about bank-related stuff is super popular in this month but not that popular later. OK? One more extension, multilingual topic models. So, topic is something that is not really dependent on the language because mathematics exists everywhere, right? So, we can just express it with different terms in English, in Italian, in Russian, and in any other language. And this model captures this intuition. So, we have some topics that are just the same for every language but they are expressed with different terms. You usually train this model on parallel data so you have two Wikipedia articles for the same topic, or let's better say for the same particular concept, and you know that the topics of these articles should be similar, but expressed with different terms, and that's okay. So, we have covered some extensions of Topic Models, and believe me there are much more in the literature. So, one natural question that you might have now if whether there is a way to combine all those requirements into one topic model. And there might be different approaches here and one approach which we develop here in our NLP Lab is called Additive Regularization for Topic Models. The idea is super simple. So, we have some likelihood for PLSA model. Now, let us have some additional regularizers. Let us add them to the likelihood with some coefficients. So, all we need is to formalize our requirements with some regularizers, and then tune those tau coefficients to say that, for example, we need better hierarchy rather than better dynamics In the model. So, just to provide one example of how those regularizers can look like, we can imagine that we need different topics in our model, so it would be great to have as different topics as possible. To do this, we can try to maximize the negative pairwise correlations between the topics. So, this is exactly what is written down in the bottom formula. You have your pairs of topics and you try to make them as different as possible. Now, how can you train this model? Well, you still can use EM algorithm. So, the E-step holds the same, exactly the same as it was for the PLSA topic model. The M-step changes, but very slightly. So, the only thing that is new here is green. This is the derivatives of the regularizers for your parameters. So, you need to add these terms here to get maximum likelihood estimations for the parameters for the M-step. And this is pretty straightforward, so you just formalize your criteria, you took the derivatives, and you could built this into your model. Now, I will just show you one more example for this. So, in many applications we need to model not only words in the texts but some additional modalities. What I mean is some metadata, some users, maybe authors of the papers, time stamps, and categories, and many other things that can go with the documents but that are not just words. Can we build somehow them into our model? We can actually use absolutely the same intuition. So, let us just, instead of one likelihood, have some weighted likelihoods. So, let us have a likelihood for every modality and let us weigh them with some modality coefficients. Now, what do we have for every modality? Actually, we have different vocabularies. So, we treat the tokens of authors modality as a separate vocabulary, so every topic will be now not only the distribution of words but the distribution over authors as well. Or if we have five modalities, every topic will be represented by five distinct distributions. One cool thing about multimodal topic models is that you represent any entities in this hidden space of topics. So, this is a way somehow to unify all the information in your model. For example, you can find what are the most probable topics for words and what are the most probable topics for time stamps, let's say. And then you can compare some time stamps and words and say, ''What are the most similar words for this day?'' And this is an example that does exactly this. So, we had a corpora that has some time stamps for the documents and we model the topics both for words and for time stamps, and we get to know that the closest words for the time stamp, which corresponds to the Oscars date would be Oscar, Birdman, and some other words that are really related to this date. So, once again, this is a way to embed all your different modalities into one space and somehow find a way to build similarities between them. OK. Now, what would be your actions if you want to build your topic models? Well, probably you need some libraries. So, BigARTM library is the implementation of the last approach that I mentioned. Gensim and MALLET implement online LDA topic model. Gensim was build for Python and MALLET is built for JAVA. And Vowpal Wabbit is the implementation of the same online LDA topic model, but it is known to be super fast. So, maybe it's also a good idea to check it out. Now, finally, just a few words about visualization of topic models. So you will never get through large collections and that is not so easy to represent the output of your model, those probability distributions, in such a way that people can understand that. So, this is an example how to visualize Phi metrics. We have words by topic's metrics here and you can see that we group those words that correspond to every certain topic together so that we can see that this blue topic is about these terms and the other one is about social networks and so on. But actually, the visualization of topic models is the whole world. So this website contains 380 ways to visualize your topic models. So, I want to end this video and ask you to just explore them maybe for a few moments, and you will get to know that topic models can build very different and colorful representations of your data.</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/xBKio/topic-models)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


4. [Week 4](https://www.coursera.org/learn/language-processing/home/week/4)

    8. Statistical Machine Translation

        * [Introduction to Machine Translation](https://www.coursera.org/learn/language-processing/lecture/nv7Cr/introduction-to-machine-translation)

<table>
  <tr>
    <td>[MUSIC] Hi everyone, this week is about
sequence to sequence tasks. We have a lot of them in NLP, but one obvious example would
be machine translation. So you have a sequence of words
in one language as an input, and you want to produce a sequence of words
in some other language as an output. Now, you can think about
some other examples. For example, summarization is also
a sequence to sequence task and you can think about it as
machine translation but for one language,
monolingual machine translation. Well we will cover these examples in the
end of the week but now let us start with statistical machine translation,
and neural machine translation. We will see that actually
there are some techniques, that are super similar in
both these approaches. For example, we will see alignments, word alignments that we need in
statistical machine translation. And then, we will see that we have
attention mechanism in neural networks that kind of has similar
meaning in these tasks. Okay, so let us begin, and
I think there is no need to tell you that machine translation is important,
we just know that. So I would better start
with two other questions. Two questions that actually we
skip a lot in our course and in some other courses but these are two
very important questions to speak about. So one question is data and
another question is evaluation. When you get some real task in your life,
some NLP task usually this is not a model that is plane,
this is usually data and evaluation. So you can have a fancy
neuro-architecture, but if you do not have good data and
if you haven't settled down how to do evaluation procedure,
you're not going to have good results. So first data, well what kind of data
do we need for machine translation? We need some parallel corpora, so
we need some text in one language and we need its translation
to another language. Where does that come from, so
what sources can you think of? Well, one of your source well maybe not so
obvious but one very good source,
is European Parliament proceedings. So you have there some texts in several
languages, maybe 20 languages and very exact translations of
one in the same statements. And this is nice, so you can use that,
some other domain would be movies. So you have subtitles that are translated
in many languages this is nice. Something which is not that useful,
but still useful, would be books translations or
Wikipedia articles. So for example, for
Wikipedia you can not guarantee that you have the same text for two languages. But you can have something similar,
for example some vague translations or which are to the same topic at least. So we call this corpora comparable but
not parallel. The OPUS website has the nice overview
of many sources so please check it out. But I want to discuss something which is
not nice, some problems with the data. Actually, we have lots of problems for
any data that we have, and what kind of problems happen for
machine translation? Well, first, usually the data
comes from some specific domain. So imagine you have movie subtitles and
you want to train a system for scientific papers translations. It's not going to work, right, so
you need to have some close domain. Or you need to know how to transfer
your knowledge from one domain to another domain,
this is something to think about. Now, you can have some decent amount of
data for some language pairs like English and French, or English and German, but
probably for some rare language pairs, you have really not a lot of data,
and that's a huge problem. Also you can have noisy and not enough
data, and it can be not aligned well. By alignment I mean, you need to know
the correspondence between the sentences, or even better the correspondence
between the words and the sentences. And this is luxury, so
usually you do not have that, at least for a huge amount of data. Okay, now I think it's clear about
the data, so the second thing, evaluation. Well you can say that we
have some parallel data. So why don't we just split it to train and
test and have our test set to compare correct translations and
those that are produced by our system. But well,
how do we know that the translation is wrong just because it doesn't
occur in your reference? You know that the language is so relative so every translator would
do some different translations. It means that if your system produce
something different it doesn't mean yet that it is wrong. So well there is no nice answer for this
question, I mean this is a problem, yes. One thing that you can do is to have
multiple references so you can have, let's say five references and
compare your system output to all of them. And the other thing is you should be
very careful how do you compare it. So definitely you shouldn't
do just exact match, right you should do
something more intelligent. And I'm going to show you BLUE score
which is known to be very popular measure in machine translation that try
somehow to softly measure whether your system output is some how similar to the reference translation. Okay, let me show you an example. So you have some reference translation and you have the output of your system and you try to compare them. Well, you remember that we have this nice tool which is called engrams. So you can compute some unigrams and bigrams and trigrams. Do you have any idea how to use that here? Well, first we can try to compute some precision, what does it mean? You look into your system output, and here you have six words, six unigrams and compute how many of them actually occur in the reference. So the unigram precision core will be 4 out of 6. Now, tell me what would be bigram score here. Well, the bigram score will be 3 out of 5 because you have 5 bigrams in your system output and only 3 of them was sent sent on and on Tuesday occurred in the reference. Now you can proceed and you can compute 3-grams score and 4-grams score, so that's good. Maybe we can just average them and have some measure. Well we could, but there is one problem here, well imagine that the system tries to be super precise. Then it is good for system to output super short sentences, right? So if I'm sure that this union gram should occur, I will just output this and I will not output more. So just to punish into penalty the model, we can have some brevity score. This brevity penalty says that we divide the length of the output by the length of the reference. And then the system outputs two short sentences, we will get to know that. Now how do we compute the BLEU score out of these values? Like this so we have some average so this root is the average of our union gram, bigram, 3-gram, and 4-gram's course. And then we multiply this average by the brevity. Okay, now let us speak about how the system actually works. So this is kind of a mandatory slide on machine translation, because kind of any tutorial on machine translation has this. So I decided not to be an exception and show you that. So the idea is like that, we have some source sentence and we want to translate it to get some target sentence. Now the first thing that we can do is just direct transfer. So we can translate this source sentence word by word and get the target sentence. But well, maybe it's not super good, right? So if you have ever studied some foreign language, you know that just by dictionary translations of every word, you usually do not get nice coherent translation. So probably we would better go into some synthetic level. So we do syntax analysis, and then we do the transfer and then we generate the target sentence by knowing how it should look like on on the syntactic level. Even better, we could try to go to semantic layer, so that first we analyze the source sentence and understand some meanings of some parts of the sentence. We somehow transfer these meanings to in our language and then we generate some good syntactic structures with good meaning. And our dream, like the best things as we could ever think of, would be having some interlingual. So by interlingual, we mean some n ice representation of the whole source sentence that is enough to generate the whole target sentence. Actually it is still a dream, so it is still a dream of the translators to have that kind of system because it sounds so appealing. But neural translation systems somehow have mechanisms that resembles that and I will show you that in a couple of slides. Okay, so for now I want to show you some brief history of the area. And like any other area, machine translation has some bright and dark periods. So in 1954 there were great expectations, so there was IBM experiments where they translated 60 sentences from Russian to English. And they said, that's easy we can solve the machine translation task completely in just three or five years. So they tried to work on that and they worked a lot, and after many years they concluded that actually it's not that easy. And they said, well, machine translation is too expensive and we should not do automatic machine translation system. We should better focus on just some tools that help human translators to provide good quality translations. So you know these great expectations and then the disappointment made the area silent for a while, but then in 1988 IBM researchers proposed word-based machine translation systems. These machine translation systems were rather simple, so we will cover them, kind of in this video and in the next video, but these systems were kind of the first working system for machine translation. So this was nice and then the next important step was phrase based machine translations system that were proposed by Philip Koehn in 2003. And this is what probably people mean by statistical machine translation now. You definitely know Google Translate, right? But maybe you haven't heard about Moses. So Moses is the system that allows a researchers to build their own machine translation systems. So it allows to train your models and to compare them, so this is a very nice tool for researchers and it was made available in 2007. Now, with an extent, obviously very important step here is neural machine translation. It is amazing how fast the neural machine translation systems could go from research papers to production. Usually we have such a big gap between these two things. But in this case there were just two or three years so it is amazing that those ideas that were proposed could be implemented and just launched in many companies in 2016 so we have neutral machine translations now. You might be wondering what is WMT there, it is the workshop on machine translation, which is kind of the annual competition, the annual event and shared tasks. Which means that you can compare your systems there, and it is a very nice venue to compare different systems by different researchers and companies. And to see what are the traits of machine translations. And it happens every year, so usually people who do research in this area keep eye on this and this is very nice thing. This is the slide about intralingual that I promised to show you. So this is how Google neural machine translation works, and there was actually lots of hype around it maybe even too much. But still, so the idea is that you train some system or some pair of languages. For example on English to Japanese and Japanese to English and English to Korean and some other pair, you train some encoder, decoder architecture. It means that you have some encoder that encodes your sentence to some hidden representation. And then you have decoder that takes that hidden representation and decodes it to the target sentence. Now the nice thing is, that if you just take your encoder, let's say for Japanese and decoder for Korean and you just take them. Somehow it works nicely even though the system has never seen Japanese to Korean translations. You see so this is zero-shot translation you have never seen Japanese to Korean, but just by building nice encoder and nice decoder, you can stack them and get this path. So it seems like this hidden representation that you have, is kind of universal for any language pair. Well, it is not completely true but at least it is very promising result.</td>
    <td></td>
  </tr>
</table>


        * [Noisy channel: said in English, received in French](https://www.coursera.org/learn/language-processing/lecture/zqKet/noisy-channel-said-in-english-received-in-french)

<table>
  <tr>
    <td>Today, we will cover one main idea of statistical machine translations. Imagine you have a sentence, let's say, in French or in some other foreign language and then, you want to have its translation to English. How do you do this? Well, you can try to compute the probability of the English sentence given your French sentence. And then, you want to maximize this probability and take the sentence that gives you this maximum probability, right? Sounds very intuitively. Now, let us apply base rule here. So let us say that instead of computing the probabilities of E given F, we would better compute probabilities of F given E. And multiply it by some probability of the English sentence. And also, normalize it by some denominator. Now, do you have any idea? Can we further simplify this formula? Well, actually, we can. So, the denominator doesn't depend on the English sentence, which means that we can just get rid of it, okay. Now, we have this formula and now, the question is, why is that easier? Why we like it more than the original formula? This slide is going to explain why. So, we have two models now. We have decoupled our complicated problem to two more simple problems. One problem is language modeling. And actually, you know a lot about it. So, this is how to produce some meaningful probability of the sentence of words. Now, the other problem is translation model. And this model doesn't think about some coherent sentences. It just thinks about some good translation of E to F, so that you do not end up with something that is not related to your source sentence. So, you have two models about language and about adequacy of the translation. And then you have argmax to perform the search in your space and find the sentence in English that gives you the best probability. Now, I have one more interpretation for you. The Noisy Channel is a super popular idea, so you definitely need to know about it. And it is actually super simple. So, you have your source sentence and you have some probability of this source sentence. And then, it goes through the noisy channel. The noisy channel is represented by the conditional probability of what you get as the output given your input for the channel. So, as the output, you obtain your French sentence. So, let's say that your source sentence was spoilt with the channel and now you obtained it in French. Now, the rest of the video is about how to model these two probabilities, the probability of the sentence and the probability of the translation given some sentence. Okay. First, about the language model. You know a lot about it so we covered this in the week two. So, I will have just one slide to have a recap for you. So, we need to compute the probability of a sentence of words. We apply chain rule and then we know that we can factorize it into the probabilities of the next word given some previous history. You can use Markov assumption and then end up with n-gram language models. Or you can use some neural language models such as LSTM to produce the next word, you will need previous words. Now, translation model. Well, it is not so easy. So, imagine you have a sequence of words in one language and you need to produce the probability of a sequence or words in some other language. For example, this is foreign language, like Russian and English language, and these two sentences. How do you produce these probabilities? Well, it is not obvious for me. So, let us start with words level. We can understand something for the level of separate words in these sentences. Okay. What can we do? We can have a translation table. So, here, I have the probabilities of Russian words given some English words. And they are normalized, right. So, each row in this matrix is normalized into one. And this are just translations that I learn or that I look up in the dictionary or built somehow. Okay, it's doable. Now, how do I build the probability of the whole sentence given these separate probabilities? We need some word alignments. So, the problem is that we can have some reorderings in the language like here, or even worse, we can have some one to many or many to one correspondence. For example, the word appetit here corresponds to the appetite. And the word with here corresponds to two Russian words [FOREIGN] It means that we need some model to build those alignments. Now, another example would be words that can appear or disappear. For example, some articles or some auxiliary words can happen in one language and then, they can't just vanish in some other language. This is a very unique word alignment models and this is the topic will fall when next video. </td>
    <td></td>
  </tr>
</table>


        * [Word Alignment Models](https://www.coursera.org/learn/language-processing/lecture/LMgKo/word-alignment-models)

<table>
  <tr>
    <td>Word Alignments Models. This is a super important subtask in machine translation, because different languages have different word order, and we need to learn that from data. So, we need to build some word alignment models and this video is exactly about them. Let us go a bit more formal and let us see what are the notations of every object that we have here. So, e is the sentence e1, e2 and so on and f is another sentence. So, the length of e sentence is I and the length of f sentence is J. Now, I need to learn some alignments between them, which I denote by a. And importantly, you'll see that I say that e is source and f is target. Why do I say so? Well, usually, we talked about machine translation system from French to English or from foreign to English. Why do I say now that it is vice versa? This is because we applied base rule. So, if you remember, we did this to have our decoupled model about language and about translation. And now, to build the system that translates from f to e, we need to model the probability of f given e. Now, what about word alignments? How do we represent them? So, the matrix of word alignments is one nice way to do this. You have one sentence and another sentence, and you have zeros or ones in the matrix. So, you'll know which words correspond to each other. Now, how many matrices do we have here? Well, actually, it is a huge amount of matrices. So, imagine you have two options in every element of the matrix and then, you have the size of the matrix which is I multiplied by J, so the number of possible matrices would be two to the power of the size of the matrix and that's a lot. So, let us do some constraints, some simplifications to deal with this. And what we do is we say that every target word is allowed to be aligned only to one source word, okay? Like here. So, this is a valid example. Now, what would be the notation here. So, we will have a1 which will represent the number of the source word which is aligned to the first target word. So, this is appetite and this is the second word. Now, what would be a2? So, a2 will be equal to three because we have comes matched to [inaudible] which is the third word in our source. Now, we can proceed and do the same for a4 and five, a6. That's it. So, this is our notation. Please keep it in mind not to get lost. Now, let us build the probabilistic model. Actually, this and the next slide will be about the sketch of the whole process. So, we are going to build the probabilistic model and figure out how we learned that. After that, we'll go into deeper details of this probabilistic model. So, stay with me. We have our sentences, e and f. So, this is our observable variables. Now, we have also word alignments. We do not see them, but we need to model them somehow. So, this is hidden variables. And we have parameters of the model and this is actually the most creative step. So, we need somehow to decide how do we parameterized our model to have some meaningful generative story. And if we have too many parameters, probably, it will be difficult to train that. If we have too less parameters, probably it will be not general enough to describe all the data. So, this is the moment that we will discuss in more details later. But for now, let's just say that we have some probabilistic model of f and a given e and theta. What do we do next? Well, you should know that in all these situations, we do likelihood maximization. So, we take our data, we write down the probability to see our data and we try to maximize this. Now, one complicated thing with this is that we do not see everything that we need to model. So, we can model the probabilities of f and a, but we don not see a. That's why we need to sum over all possible word alignments. And on the left-hand side, you have the probability of f given all the rest things, which is called incomplete data. Likelihood maximization for incomplete data means that there are some hidden variables that you do not see. And this is a very bad situation. So, imagine you have a logarithm. So, you take logarithm and you have logarithm of the sum. And you don't know how to maximize these, how to take derivatives and how to get your maximum likelihood estimations. But actually, we have already seen this case two times in our course. So, one was Hidden Markov Model and another was topic models. In both those cases, we had some hidden variables and we have these incomplete data. And in both cases we used EM-algorithm. So, EM-algorithm just to recap, is an iterative process that has E-step and M-step. The E-step is about estimates for your hidden variables. So, the E-step will be, what are the best alignments that we can produce right now given our perimeters? And the M-step is vice versa. Given our best guess for word alignments, what would be the updates for parameters that maximize our likelihood? This is also so interesting to go into the exact formulas of EM-algorithm. Better, let us discuss generative model because it is really creative thing. Well, let us start with generating the length of the sentence. So, J would be the length of the target sentence. Once we could generate this, let us say that we have independent susception by the target words. So, we have this product by J which denotes the word in our target sentence. Every word will be not modeled yet. So first, real model the alignment for every position. And then, we will model the exact word given that alignment. So, if you are scared with this formula, you can look into just green parts. This is the most important thing. You model alignments and you model words given these alignments. All the other things that you see on the right would be just everything that we know to condition on that. And this is too much to condition on that because we will have well too much parameters. So, we need to do some assumptions. So, we need to say that not all those things are important in this conditions. The first IBM model is the first attempt to simplify this generative story. So, what it says is, let us forget about the priors for word alignments, let us have just a uniform prior. And this prior will know nothing about the positions, but it will have just one constant to tune. So, this is awesome. Now, the translation table will be also very simple. So, we will say that the probability to generate the word, given that it is aligned to some source word, is just the translation probability of that word given the source word. So, how does that look like? This is the translation table. So, once we know that the word is aligned to some source word, we just take this probability out of it. So, this is a very simple model, but it has a very big drawback. It doesn't take into account the position of your word to produce the alignment to this word. So, the second IBM model tries to make better and it says, "Okay, let us take J, the position of the target word and let us use it to produce aj.", the alignment for this target word. Now, how many parameters do we have to learn to build this position-based prior. Well, actually a lot of parameters. So, you know what, you have I multiplied by J, which will be the size of the matrix of probabilities and it is not all. Apart of this matrix, you will also have different matrices for different I and J. So, you cannot just use one and the same matrix for all kind of sentences. You just share this matrix across all sentences with given lengths. But for sentences with different lengths, you have different matrix. So, this is a lot of parameters and to try to improve on that, we can say, "Well, the matrix is usually very close to diagonal. What if we model it as a diagonal matrix?" This is what Chris Dyer said in 2013. So, this model has only one perimeter that says, how is the probability mass spread around the diagonal? And this is nice because it is still has some priors about positions, but it has not too many parameters. Now, I have the last example for you for alignments. So, actually, you already know how to build this, you just don't remember that. We had Hidden Markov Models in our course and Hidden Markov Models can help to build some transition probabilities. Why do we need it here? So, imagine these couple of sentences and the phrase in the beginning of the sentence can be aligned to the phrase in the end of the sentence. But sometimes, inside this phrase, you just go word-by-word so you do not have any gaps. And this is nice. It means that you need to learn these and to use this information that the previous word was aligned to position five and maybe that means that the next word should be aligned to position six. So, this is what Hidden Markov Model can make for you. So, you model the probability of the next alignment given the previous alignment. So now, let us sum up what we have in this video. So, we have covered IBM models, which is a nice word-based technique to build machine translation systems. And actually, there are lots of problems with them that we did not cover. And there are IBM Model Number three and four and five that can try to cope with the problem of fertility, for example, saying that we need to explicitly model how many output words are produced by each source word, or that we need to explicitly deal with spurious word. This are the words that just appear from nowhere in the target language</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/CVBTj/introduction-to-machine-translation)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


    9. Encoder-decoder-attention arhitecture

        * [Encoder-decoder architecture](https://www.coursera.org/learn/language-processing/lecture/bGV7m/encoder-decoder-architecture)

<table>
  <tr>
    <td>[SOUND] Hey everyone,
we're going to discuss a very important technique
in neural networks. We are going to speak about
encoder-decoder architecture and about attention mechanism. We will cover them by the example
of neural machine translation, just because they were mostly proposed for
machine translation originally. But now they are applied to many,
many other tasks. For example, you can think about
summarization or simplification of the texts, or sequence to sequence
chatbots and many, many others. Now let us start with the general
idea of the architecture. We have some sequence as the input, and we would want to get some
sequence as the output. For example, this could be two sequences
for different languages, right? We have our encoder and the task of
the encoder is to build some hidden representation over the input
sentence in some hidden way. So we get this green hidden vector that tries to encode the whole
meaning of the input sentence. Sometimes this vector is
also called thought vector, because it encodes
the thought of the sentence. The encoder task is to decode
this thought vector or context vector into some
output representation. For example, the sequence of
words from the other language. Now what types of encoders
could we have here? Well, one most obvious type would
be her current neural networks, but actually this is not the only option. So be aware that we have also
convolutional neural networks that can be very fast and nice, and they can also
encode the meaning of the sentence. We could also have some
hierarchical structures. For example, recursive neural networks
try to use syntax of the language and build the representation hierarchically
from from bottom to the top, and understand the sentence that way. Okay, now what is the first example
of sequence to sequence architecture? This is the model that was proposed
in 2014 and it is rather simple. So it says, we have some LCM module or RNN
module that encodes our input sentence, and then we have end of
sentence token at some point. At this point, we understand that
our state is our thought vector or context vector, and we need to
decode starting from this moment. The decoding is conditional
language modelling. So you're already familiar with language
modelling with neural networks, but now it is conditioned on this
context vector, the green vector. Okay, as any other language model,
you usually fit the output of the previous state as the input to the next state, and
generate the next words just one by one. Now, let us go deeper and
stack several layers of our LSTM model. You can do this
straightforwardly like this. So let us move forward, and speak about a little bit different
variant of the same architecture. One problem with
the previous architectures is that the green context
letter can be forgotten. So if you only feed it as the inputs
to the first state of the decoder, then you are likely to forget about it when you
come to the end of your output sentence. So it would be better to
feed it at every moment. And this architecture does exactly that,
it says that every stage of the decoder should have three
kind of errors that go to it. First, the error from the previous state,
then the error from this context vector, and then the current input which is
the output of the previous state. Okay, now let us go into more
details with the formulas. So you have your sequence
modeling task conditional because you need to produce the probabilities of
one sequence given another sequence, and you factorize it using the chain rule. Also importantly you see that
x variables are not needed anymore because you have
encoded them to the v vector. V vector is obtained as the last
hidden state of the encoder, and encoder is just recurrent neural network. The decoder is also
the recurrent neural network. However, it has more inputs, right? So you see that now I concatenate
the current input Y with the V vector. And this means that I will
use all kind of information, all those three errors in my transitions. Now, how do we get predictions
out of this model? Well, the easiest way is just
to do soft marks, right? So when you have your decoder RNN, you have your hidden states of
your RNN and they are called SJ. You can just apply some linear layer,
and then softmax, to get the probability of the current
word, given everything that we have, awesome. Now let us try to see whether those
v vectors are somehow meaningful. One way to do this is to say, okay they are let's say three
dimensional hidden vectors. Let us do some dimensional reduction,
for example, by TS&E or PCA, and let us plot them just by two dimensions
just to see what are the vectors. So you see that the representations
of some sentences are close here and it's nice that the model
can capture that active and passive voice doesn't actually matter for
the meaning of the sentence. For example, you see that the sentence,
I gave her a card or she was given a card are very
close in this space. Okay, even though these representations are so nice, this is still a bottleneck. So you should think about how to avoid that. And to avoid that, we will go into attention mechanisms and this will be the topic of our next video.</td>
    <td></td>
  </tr>
</table>


        * [Attention mechanism](https://www.coursera.org/learn/language-processing/lecture/1nQaG/attention-mechanism)

<table>
  <tr>
    <td>Hey. Attention mechanism is a super powerful technique in neural networks. So let us cover it first with some pictures and then with some formulas. Just to recap, we have some encoder that has h states and decoder that has some s states. Now, let us imagine that we want to produce the next decoder state. So we want to compute sj. How can we do this? In the previous video, we just used the v vector, which was the information about the whole encoded input sentence. And instead of that, we could do something better. We can look into all states of the encoder with some weights. So this alphas denote some weights that will say us whether it is important to look there or here. How can we compute this alphas? Well, we want them to be probabilities, and also, we want them to capture some similarity between our current moment in the decoder and different moments in the encoder. This way, we'll look into more similar places, and they will give us the most important information to go next with our decoding. If we speak about the same thing with the formulas, we will say that, now, instead of just one v vector that we had before, we will have vj, which is different for different positions of the decoder. And this vj vector will be computed as an average of encoder states. And the weights will be computed as soft marks because they need to be probabilities. And this soft marks will be applied to similarities of encoder and decoder states. Now, do you have any ideas how to compute those similarities? I have a few. So papers actually have tried lots and lots of different options, and there are just three options for you to try to memorize. Maybe the easiest option is in the bottom. Let us just do dot product of encoder and decoder states. It will give us some understanding of their similarity. Another way is to say, maybe we need some weights there, some metrics that we need to learn, and it can help us to capture the similarity better. This thing is called multiplicative attention. And maybe we just do not want to care at all with our mind how to compute it. We just want to say, "Well, neural network is something intelligent. Please do it for us." And then we just take one layer over neural network and say that it needs to predict these similarities. So you see there that you have h and s multiplied by some matrices and summed. That's why it is called additive attention. And then you have some non-linearity applied to this. These are three options, and you can have also many more options. Now, let us put all the things together, just again to understand how does attention works. You have your conditional language modeling task. You'll try to predict Y sequence given s sequence. And now, you encode your x sequence to some vj vector, which is different for every position. This vj vector is used in the decoder. It is concatenated with the current input of the decoder. And this way, the decoder is aware of all the information that it needs, the previous state, the current input, and now, this specific context vector, computed especially for this current state. Now, let us see where the attention works. So neural machine translation had lots of problems with long sentences. You can see that blue score for long sentences is really lower, though it is really okay for short ones. Neural machine translation with attention can solve this problem, and it performs really nice for even long sentences. Well, this is really intuitive because attention helps to focus on different parts of the sentence when you do your predictions. And for long sentences, it is really important because, otherwise, you have to encode the whole sentence into just one vector, and this is obviously not enough. Now, to better understand those alpha IJ ways that we have learned with the attention, let us try to visualize them. This weights can be visualized with I by J matrices. Let's say, what is the best promising place in the encoder for every place in the decoder? So with the light dot here, you can see those words that are aligned. So you see this is a very close analogy to word alignments that we have covered before. We just learn that these words are somehow similar, relevant, and we should look into this once to translate them to another language. And this is also a good place to note that we can use some techniques from traditional methods, from words alignments and incorporate them to neural machine translation. For example, priors for words alignments can really help here for neural machine translation. Now, do you think that this attention technique is really similar to how humans translate real sentences? I mean, humans also look into some places and then translate this places. They have some attention. Do you see any differences? Well, actually there is one important difference here. So humans save time with attention because they look only to those places that are relevant. On the contrary, here, we waste time because to guess what is the most relevant place, we first need to check out all the places and compute similarities for the whole encoder states. And then just say, "Okay, this piece of the encoder is the most meaningful." Now, the last story for this video is how to make this attention save time, not waste time. It is called local attention, and the idea is rather simple. We say, let us first time try to predict what is the best place to look at. And then after that, we will look only into some window around this place. And we will not compute similarities for the whole sequence. Now, first, how you can predict the best place. One easy way would be to say, "You know what? Those matrices should be strictly diagonal, and the place for position J should be J." Well, for some languages, it might be really bad if you have some different orders and then you can try to predict it. How do you do this? You have this sigmoid for something complicated. This sigmoid gives you probability between zero to one. And then you scale this by the length of the input sentence I. So you see that this will be indeed something in between zero and I, which means that you will get some position in the input sentence. Now, what is inside those sigmoid? Well, you see a current decoder state sj, and you just apply some transformations as usual in neural networks. Anyway, so when you have this aj position, you can just see that you need to look only into this window and compute similarities for attention alphas as usual, or you can also try to use some Gaussian to say that actually those words that are in the middle of the window are even more important. So you can just multiply some Gaussian priors by those alpha weights that we were computing before. Now, I want to show you the comparison of different methods. You can see here that we have global attention and local attention. And for local attention, we have monotonic predictions and predictive approach. And the last one performs the best. Do you remember what is inside the brackets here? These are different ways to compute similarities for attention weights. So you remember dot product and multiplicative attention? And, also, you could have location-based attention, which is even more simple. It says that we should just take sj and use it to compute those weights. This is all for that presentation, and I am looking forward to see you in the next one. Downloads Lecture Videomp4 Subtitles (English) WebVTT Transcript (English) txt Slides pdf Would you like to help us translate the transcript and subtitles into additional languages?</td>
    <td></td>
  </tr>
</table>


        * [How to deal with a vocabulary?](https://www.coursera.org/learn/language-processing/lecture/mvV6t/how-to-deal-with-a-vocabulary)

<table>
  <tr>
    <td>[SOUND] This video is about all kind
of problems that you can have with vocabulary in machine translations. So first, vocabulary is usually too large,
and it is too long to compute softmax. Second, vocabulary is usually too small,
and you have out-of-vocabulary words, and
you need somehow to deal with them. And you have a whole range
on different tricks for you in neural language modeling and
in machine translation. Now, let us start with
hierarchical softmax. It is a nice procedure to help you
to compute softmax in a fast way. So the idea is to build the binary
tree for your words, like here. And the words in this tree
will have some codes. So for example, the zebra will have
code 01, which means that first, we go to the left, it is 0, and
then we go to the right, it is 1. And importantly, it is just unique mapping
between words and codes in the tree. So we are going to use this property,
and we are going to produce the probabilities for
words using their binary codes. This is how we make it. So instead of computing the probability
of the word and normalizing it across the whole vocabulary, we are going
to split it into separate terms. Each term is the probability of the digit,
is the probability of the next decision in the tree,
whether to go to the left or to the right. And we build these probabilities
into the product so that this product is going to estimate
the probability of the whole word. Now, do you believe that this
product of binary probabilities is going to be normalized into
one across the whole vocabulary? Well, sounds like a magic, isn't it? But let's see that actually, it happens. So you see that this is your binary tree,
and you see those probabilities. They are just written down for
every path in the tree. Now, you can try to sum
the first two probabilities, and you note that 0.1 plus 0.9 gives 1,
and this is not just random. This is always 1,
just because this is probability for two options, right, going to the left and
going to the right. That's why they
are necessary summed into 1. So if you sum those things, you get their
common prefix, which will be 0.7 by 0.8. Then you can try to sum two
other childs in the tree, and you get, again, their common prefix. You sum two other childs, and
you get, again, their common prefix. And finally, you sum these two values,
and you get 1. So it is clearly seen that
if you go from bottom to the top with this tree,
you will get probability equal to 1. So just to sum it up,
we split the probability into these binary solutions, and
we use some tree to do this. What kind of tree do we do? Well, it is actually not so important. So interestingly,
even just random tree can help. Alternatively, you can use
Huffman tree that gets to know the frequencies of some words,
and it uses that. Or you can try to use some semantics so you can just cluster your words in
the data based on their similarities. Or you can use some pre-built structure
that says that these words are similar and that they should be found
in one hierarchy branch. Okay, now I have one last question for
you for this method. What would be a way to
model the probability of di given all the other stuff? So usually, we have softmax
to model some probabilities. What would be the probability
in this case? Well, in this case, we have just
only two options, left and right. So instead of softmax, we can have sigmoid
function, which has just two outcomes. Okay, now let us come back to some
other problems with our bot vocabulary. And copy mechanism is something to
help with out-of-vocabulary words. Imagine you have some sentence,
and some words are UNK tokens. So you do not have them in the vocabulary,
and you do not know how to translate them, you get UNK tokens as the result. But what if you know the word alignments? If you know how the words are aligned,
you can use that. So you can say, okay, this UNK corresponds to that UNK,
which corresponds to this source word. Let us just do dictionary translation, or let us just copy, because this is the name
of some place or some other name. Let us just copy this as is. This is why it is called copy mechanism,
and the algorithm is super simple. So you need first to make sure that
you somehow learn word alignments. For example, your neural machine translation system
has these alignments as an input. And it tries to predict them along
with the translation predictions. Now, you get your translation with UNK
tokens, and you post-process this. So you just copy the source words, or you translate them with dictionary or
do whatever else what you want. Okay, very simple and nice technique, but
actually, there are still many problems. For example,
you can have some multi-word alignments. What if the morphology of
the languages are complicated, and probably, you want to split
it somehow into some parts? Or what if you have
some informal spelling? All those things are usually
out-of-vocabulary words. And these examples show
you that sometimes, it is very nice to go to sub-word level. For example, for rich morphology, it would be nice to model every piece of the word independently. Or for informal spelling, it would be definitely good to model them by letters. Because there is no chance to find these words as a whole in the vocabulary. Okay, so there are two big trends in sub-word modelling. One big trend is to do some hybrid models that somehow combines word-level models and character-level models. Another big trend is to do the same architecture, let's say, recurrent neural network, but with some small units, something in between words and characters. So this is one architecture, but other units. Okay, let us start with hybrid models. So you might know that sometimes, character-based models are super useful. For example, you see the word drinkable. And if you can build your convolutional neural network that has some convolutions that represent the meaning of drink. And then some other convolutions that represent the meaning of able, you are likely to build the meaning of the whole word, even if you have never seen that. So character-level convolutional neural networks can work fine. Also, bidirectional LSTMs can be used on word level as well as on character level. Now, for our translation model, it is super nice to have hybrid models. Let's say, let us first try to work on word level. So let us try to produce word translations. For example, we have a cute cat here. And a and cat are nice words. But what if cute is out of vocabulary? We cannot model it on word level. In this case, let us have some separate units, some separate architecture, maybe even completely different, that will model these probabilities for this word on character level, and the same for the decoder. So first, we will try to decode the sequence on word level. And then, in some moments, the decoder will say, okay, this is some UNK token, please do something about it. And then the character-level model will switch on and do something about it. So this is a very nice and obvious idea, and it is used a lot. Now, the second big trend would be sub-word modeling, and one good example of that is byte-pair encoding. Let us understand what it is. So imagine you have some sentence, and you want to encode this, and you are not quite sure yet what is your vocabulary. You have just some constraints for the size of the vocabulary. So you start with characters. Everything is split to single characters. Then you say, okay, what are the most popular bigrams of my letters here? Well, I see that S-H happens three times. So maybe I should collapse them into just one unit. And you do this, and you have some other vocabulary right now. Okay, now you say, okay, what is next? Next, I see that these two letters occur a lot, let us collapse them. These two letters also should be collapsed. And then importantly, you can apply the same procedure to sub-word units. So here, you would collapse your bigrams and unigrams into trigrams. And actually, you can stop whenever you want. So you can proceed until you get the nice size of the vocabulary that you like. Yep, I have just said you this. So one thing to mention is how to apply this method for test data. So if you have test data, you also split it to letters first. And then you know the exact rules from your training procedure. And you apply those rules to test data, to collapse all the sub-word units as needed. Awesome, so this chart shows you why it is so cool technique. So this is the vocabulary size, and this line there, this vertical line, is about the size of the vocabulary that we are allowed. And in case of words, usually you have some long tail that goes outside of this allowed amount of words. But with byte-pair encoding, you can do exactly this number of units, because you decide which rules to apply. And finally, I can show you that this actually works fine. So for some different pairs of languages, for some different tasks from, for example, WMT, which is Workshop on Machine Translation, you can see that byte-pair encoding has better BLEU score than word-based techniques. And actually, this BLEU score improvement is very meaningful. This one or two points of BLEU is a lot. So please use this very nice technique if you someday need to build machine translation system</td>
    <td></td>
  </tr>
</table>


        * [How to implement a conversational chat-bot?](https://www.coursera.org/learn/language-processing/lecture/AlMoB/how-to-implement-a-conversational-chat-bot)

<table>
  <tr>
    <td>[MUSIC] Hey, in this video we are going
to discuss how to apply encode or decode attention architecture to
a hot topic nowadays, to chat-bots. First of all, let us understand
what do we mean by chat-bots? As for any hot topic, every one
means something slightly different. So first we can have goal-oriented bots,
and just bots that have a nice
conversation with us. So for goal-oriented bots, we usually
speak about some narrow domain and specific tasks to fulfill. For example, this can be a bot
in a call center in our bank that can help customers with their needs. Worth is what can serve some specific answers to some specific
questions of the customers. It means that usually these bots
are based on some retrieval approach, for example, we could have some
database of information, and then it'll just solve those answers for
the users. Now, entertaining bots or bots that can
just have some conversation with us. I usually call chit chat bots, and
the model for them would be generative. What do I mean by generative models.? It means that we can just
generate new responses. Just to compare for
retrieval-based models, we would get some predefined responses
just ranked from some repository. Or we would have some patterns,
and then we would just use these patterns to get
some specific replies. Now, there are some pros and
cons of two models, obviously for generated models you have more freedom,
you can generate whatever you want. But you can also have some mistake, and it
is more complicated to build these models. So in this video, we'll speak only
about what generative models for conversational chat-bots. And those bots that have some goals to
assist users with their specific needs, will be more covered in the next week. Okay, so let us just recap that we could
have our encoder to encode the incoming message for the bot, and decoder that
would generate the bot's response. And we can have, for example, LSTMs in encoder and decoder, and
we would also want to have attention. One other alternative for
attention would be to have reversed input. Actually, it is a very simple thing
that had been studies before attention. So, we shall say, maybe we need to
reverse the input of the sentence, and then our thought vector of the
sentence will be somehow more meaningful. Why? To understand that, let us cover one
technical detail that we need to understand to built any
system of that kind. The detail is about a set effect that all
the sequences have different lengths. Okay, so you have some questions,
for example, and you need to somehow pad those questions to have some
fixed length for all of them. Why? Just because you need to
pack them into batches and send these batches to your
neural network for your. So it suggests some technical
implementation detail that if you implement as static
recurrent neural network, you need to pad your sequences
with some pad tokens. So if you do this in
the end of the sequence, you could see that the end the sequence
is absolutely not meaningful, right? It is just PAD, PAD, PAD, and so on. So if you try to build your thought vector
base on it, maybe it will not be nice. That's where you just reverse everything,
and then you have your words
in the end of the sequence. Now you can encode to that, and the decoder will get some answer for
you also pad it as news. One other idea would be to do bucketing. What it means is that let us
group our sentences into buckets, based on their length. For example, those sentences that
have the length less than five, would go into the first bucket. And then for them, we can just
pad them to the length of five. So this approach will give
us an opportunity to have not that many pad tokens. Because we will have the adaptable length, based on the maximum
length in each bucket. The only important thing is to put these
buckets into different batches then, just to make sure that the recurrent
neural network will not get buckets with a different lengths
in one of the same batch. Okay, in the rest of the video, I'm going
to show you how the chat-bots work, but we will also discuss
how they do not work. So you'll see some analysis
over the problems, and just a very ideas of what can be fixed. So, you can see that this is
a human tool machine talk, trained for movie subtitles,
and it is rather impressive. So, what is the purpose of living? And the machine says,
to live forever, okay sounds good. But you can also notice that it
is very dependent on the data set that we use to train the model. So if we try to use the same model,
lets see for assistant in a bank, maybe that's not a good idea. So there responses will be too dramatic
and the topics might be unrealistic. So it is very important to understand
that you have some specific properties of the outcome,
based on the domain of the data. Now, if you want to use it for
calls, let us train it on calls. So you have some meaningful lexis here,
but it is inconsistent. So the chat-bot says, what is
the operating system of your machine? And the user says linux. After that, just in a few turns, the machine says again, so is it a windows machine? And the user has to answer again. So this is not nice, the bot doesn't remember what was happening before in our conversation, and you can try to fix that. So there is a paper that says, you would need to track the intent and the context of the conversation with some separate recurrent neural network. And then we can somehow just memorize for the bot that these topics have been already covered. And we do not need to ask again what is the operating system of the machine, let's say. So you do not see such problems for this example of the dialogue. Now another important problem is that the bot has no personality. So if you try to ask the bot, where do you live now? The bot can say, I live in Los Angeles, and that sounds okay. But then if you ask the bot again you will get some other responses. Just because it was trained on the data of questions and answers, and the bot doesn't know about consistency in them. So, one idea would be to build persona-based models. It means that, we need to memorize that the bot has some personality. So we just train iton some coherent pieces of dialogues from different persons, and we built this knowledge of persons. So that when we ask what is the country, what is the city, we still get coherent responses. Now, another problem is diversity in the responses. So this is a smart reply technology by Google, that says that it can help you to answer Gmail automatically. For example, if you see the email, let us meet up and discuss, you want to get some proposed responses. And the model would propose how about tomorrow, what about tomorrow, I suggest tomorrow, so there is no diversity. And the user cannot pick one of them because all of them are about the same thing. Also another problem would be that you have two popular responses that can come to any email. Again, you will all have not enough diversity and you will have I love you, even before some email from a colleague that is not good for your chat-boat. So how can we cope with that? Do you have any idea how may be to track that? One idea would be to do intent clustering for our responses. For example, we can have some small supervised data, about the types of the responses. For example, you can have the label, how about time for how to how about Friday or something like that. So you have actually some graph of different responses, and you have similarities between the responses built by some bearings, or some distributional semantics model. Now, you have some labeled nodes in this graph from your supervised data, and you want to propagate this knowledge to other labels of the graph. So this technique is called label propagation on graphs, and Expander is a library that implements this technique. So the main idea here is that we will try to propagate the labels of the responses, in such a way that close responses will get close labels. And those in such a way that those labels that already are known from our supervision, will be stayed the same, awesome. So the methods can be different but the idea is just to do something clustering, and then to pick up one example from every cluster, and suggest it for the user. So what we get out of it is very nice, so this is the query and this is the top generated responses. And you see that now we have how about Tuesday? I can on Wednesday, I can on some other day, so you see that you have some diversity, and that's what we want you to have. Well, even though the bots can try to have some meaningful conversation with you, you can see that there are still so many problems with them. And it is so easy to understand that you are speaking to a bot and not to a human. And that's why actually we should be very careful about that hype. And we should realize that well, indeed, we are very promising and we have some good opportunities in the future, but current models are still not humans</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/AjG2D/encoder-decoder-architectures)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


        * [Peer-graded Assignment: Learn to calculate with seq2seq model](https://www.coursera.org/learn/language-processing/peer/3wvhr/learn-to-calculate-with-seq2seq-model)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


    10. Summarization and simplification tasks

        * [Sequence to sequence learning: one-size fits all?](https://www.coursera.org/learn/language-processing/lecture/4z2ox/sequence-to-sequence-learning-one-size-fits-all)

<table>
  <tr>
    <td>[MUSIC] Hey, let us see how many different
tasks in NLP can be solved as sequence to sequence tasks. So we have talked a lot about machine
translation, that's obvious, but also you have so many other options. For example, you can do speech recognition
and there is model called listen, attend and spell. Or you can do image caption generation,
And this will be also in quarter to quarter architecture and
the paper is called show, attend and tell. So they are so similar, however every
task can be solved specifically better if you just think a little bit about those
constraints that you have in this task. So in this video, we'll speak in more
details about text simplification. And we will see that, well, we can use
just in quarter to quarter architecture. But if we think a little bit
about the specific objections for this task, we can improve. Okay, let us start with summarization for
now. Summarization task is when you need to
get the short summary of some document. Summarization can have several types. So we can speak about
extractive summarization, and it means that we just extract
some pieces of the original text. Or we can speak about
abstractive summarization, and it means that we want to generate
some summary that is not necessarily from out text, but
that nicely summarizes the whole text. So this is obviously better, but
this is obviously more difficult. So most production systems would
be just some extractive summaries. Now, let us try to do extractive summary
with sequence to sequence model. So you get your full text as the input,
and you get your summary as the output, and you have your encoder and
decoder as usual. Now, you need some good
dataset to train your model. For example, English Gigaword
dataset is really huge, and it contains some examples of articles and
their headlines. Now you apply your model, and there is
also even open-source implementation for the model, so you can just use it
right away and get some results. So the results are rather promising. You can see the sentence of the article, just the first sentence,
and the generated headline. So the thing on the right is
just generated by our model. Actually, there are some
problems with this model, and we will speak about them in
another video in our course. But for now,
we can just say that it works somehow. And let us move forward and discuss another very related task,
which is called simplification. Text simplification task would also
need some good dataset to train on. And one good candidate
would be simple Wikipedia. So you see that you have some
normal Wikipedia sentences and simple Wikipedia sentences,
what can be different there? For example, you can have some deletions. For example, in the second example,
you just delete two pieces, and in the first example you try
to rephrase some pieces. What kind of operations you can
have to modify these sentences? Well, as I have already said,
you can delete, you can paraphrase, or you can just split one sentence
to two simpler smaller sentences. Now, paraphrasing is rather general
approach and you can do different things. You can reorder words, or you can do some
syntactic analysis of your sentence and understand that some syntactic
structures are more simple and usual just substitute one syntactic
structures by some others. And the straight forward way to do
this would be rule-based approach. Actually, we do not cover ruled-based
approach a lot in our course, well maybe it's not so
fancy as deep neural networks. So usually people want to hear about
deep neural networks more, but to be honest, rule-based approach is a
very good thing that works in production. So if we just want to be sure that
your model performs okay, it's a good idea to start with just implementing
your specific rules for the model. For text simplification task,
it can be just some substitutions, some context free grammar
rules that tell you that, for example, solely can be simplified by only. Or if you say something population,
you should better say, the population of something, okay? So lots of rules,
you can either know them, for example if you have some linguists,
or you can learn them from data. So this paraphrase database
is a big data source, and it also has some learned rules. Another approach would be still
to do some deep learning and even reinforcement learning. So this is not easy to make that model
work, but I just want to give you some general idea, very hand-wavey
idea of how it could be done. You can do just encoder-decoder
architecture as usual. But this architecture is
likely not to simplify well, because it doesn't have any
simplification objective built in. So one way to build in this objective would be weak supervision
by reinforcement learning. What do I mean by that? In reinforcement learning, we usually have
some agents that perform some actions. So here, the actions would be
to generate the next word. Usually we also have some policy, which
means the probability distribution for actions. And in this case, it will be probabilities of the next word given everything else. And the agent performs some actions according to the policy and gets some rewards. So if the generated sentence is good, then the reward should be high. So one very creative step is how do we estimate this reward? And the idea is to do it in three parts. So adequacy is about whether the simplified sentence is still about the same fact as the original sentence. Fluency is just about the coherence of the sentence and the language model. And simplicity is whether the simplified version is indeed simpler than the original one. A super high level architecture would be as follows. You have your encoder-decoder agent that can generate some sequences. Then for every sequence you get some rewards based on simplicity, relevance and fluency. These rewards go to reinforce algorithm that we do not cover right now, but you need to know that this reinforced algorithm can use these rewards to update the policy of the agent. So the agent, on the next step, will be more likely to generate those actions that give higher rewards. So in some sense it is similar to gradient descent, you would say, but the important distinction is that the rewards are usually not differentiable. So reinforcement learning is really helpful when you cannot just say that you have your most function and you need to optimize it. But when you just say, well, this is simple, this is not simple, so here the reward is high, here the reward is low. If the reward is like that, you cannot just take gradients and do stochastic gradient descent. And that's why you apply something a little bit more magical, which is called a reinforced algorithm. Now, I just want go into details of just one piece on this slide, simplicity. So how do we measure simplicity? Well, we have three kinds of information. Input, which is the normal text, then references, which is the golden simplified sentences, and then our output over the system. We need to compare all of them to understand whether we perform well. For example for machine translation, you would compare just human references with system outputs, right? Because the input is usually in some other language. But here it is very important to compare all of them. For example, one nice measure that can be used is called system against reference and input. It computes precision course for different types of operations, for addition, copying, and deletion. For example, what would be the precision for addition operation? Well, what are those terms that we add? These are the terms that occur in output, but do not occur in input. And this is exactly what we see in the denominator. Now, how many of them occur in the reference? This is exactly what we see in the nominator. So we just have precision score that measures how many of the terms are indeed correct. Now you can think about recall for addition, and precision and recall for other operations, and somehow average them to get this score. I want to show you that this score actually works. For example, we have the input and three references and three outputs. And you can see that the second output is definitely better than the third one, because now is simplified, we had currently in the input. And this score can distinguish this, because we compare everything with the input. It doesn't happen for BLEU score for machine translation. There, we compare just output and reference. And the BLEU score thinks that system number two and system number three behaves just the same. </td>
    <td></td>
  </tr>
</table>


        * [Get to the point! Summarization with pointer-generator networks](https://www.coursera.org/learn/language-processing/lecture/RhxPO/get-to-the-point-summarization-with-pointer-generator-networks)

<table>
  <tr>
    <td>Hey, in this video, I'm going to cover one nice paper about summarization. This is a very recent paper from Chris Manning Group, and it is nice because it tells us that on the one hand, we can use encoder-decoder architecture, and it will work somehow. On the other hand, we can think a little bit and improve a lot. So, the improvement will be based on pointer networks, which are also a very useful tool to be aware of. Also sometimes, we have rather hand-wavy explanations of the architectures with the pictures. Sometimes, it is good to go into details and to see some actual formulas. That's why I want to be very precise in this video, and in the end of this video, you will be able to understand all the details of the architecture. So, this is just a recap, first of all that we have usually some encoder, for example bidirectional LSTM and then we have some attention mechanism, which means that we produce some probabilities that tells us what are the most important moments in our input sentence. Now, you see there is some arrow on the right of the slide. Do you have any idea what does this arrow means? Where does it comes from? Well, the attention mechanism is about the important moments of the encoder based on the current moment of the decoder. So, now we definitely have the yellow part which is decoder, and then the current state of this decoder tells us how to compute attention. Just to have the complete scheme, we can say that we use this attention mechanism to generate our distribution or vocabulary. Awesome. So, this is just a recap of encoder-decoder attention architecture. Let us see how it works. So, we have some sentences, and we try to get a summary. So, the summary would be like that. First, we see some UNK tokens because the vocabulary is not big enough. Then, we also have some problems in this paragraph that we will try to improve. One problem is that the model is abstractive, so the model generates a lot, but it doesn't know that sometimes, it will be better just to copy something from the input. So, the next architecture will tell us how to do it. Let us have a closer look into the formulas and then see how we can improve the model. So, first, attention distribution. Do you remember notation? Do you remember what is H and what is S? Well, H is the encoder states and S is the decoder states. So, we use both of them to compute the attention weights, and we apply softmax to get probabilities. Then, we use these probabilities to weigh encoder states and get v_j. v_j is the context vector specific for the position j over the decoder. Then how do we use it? We have seen in some other videos that we can use it to compute the next state of the decoder. In this model, we will go in a little bit more simple way. Our decoder will be just normal RNN model, but we will take the state of this RNN model s_j and concatenate with v_j and use it to produce the probabilities of the outcomes. So, we just concatenate them, apply some transformations, and do softmax to get the probabilities of the words in our vocabulary. Now, how can we improve our model? We would want to have some copy distribution. So, this distribution should tell us that sometimes it is nice just to copy something from the input. How can we do this? Well, we have attention distribution that already have the probabilities of different moments in the input. What if we just sum them by the words? So, for example, we have seen as two times in our input sequence. Let us say the probability of as should be equal to the sum of those two. And in this way, we'll get some distribution over words that occurred in our input. Now, the final thing to do will be just to have a mixture of those two distributions. So, one is this copy distribution that tells that some words from the input are good, and another distribution is our generative model that we have discussed before. So just a little bit more formulas. How do we weigh these two distributions? We weigh them with some probability p generation here, which is also sum function. So every thing which is in green on this slide is some parameters. So, you just learn these parameters and you learn to produce this probability to weigh two kinds of distributions. And this weighting coefficient depends on everything that you have, on the context vector v_j, on the decoder state s_j, on the current inputs to the decoder. So you just apply transformations to everything that you have and then sigmoid to get probability. The training objective for our model would be, as usual, cross-entropy loss with this final distribution. So, we will try to predict those words that we need to predict. This is similar to likelihood maximization, and we will need to optimize the subjective. Now, this is just the whole architecture, just once again. We have encoder with attention, we have yellow decoder, and then we have two kinds of distributions that we weigh together and get the final distribution on top. Let us see how it works. This is called pointer-generation model because it has two pieces, generative model and pointer network. So this part about copying some phrases from the input would be called pointer network here. Now, you see that we are good, so we can learn to extract some pieces from the text, but there is one drawback here. So you see that the model repeats some sentences or some pieces of sentences. We need one more trick here, and the trick will be called coverage mechanism. Remember you have attention probabilities. You know how much attention you give to every distinct piece of the input. Now, let us just accumulate it. So at every step, we are going to sum all those attention distributions to some coverage vector, and this coverage vector will know that certain pieces have been attended already many times. How do you compute the attention then? Well, to compute attention, you would also need to take into account the coverage vector. So the only difference here is that you have one more term there, the coverage vector multiplied by some parameters, green as usual, and this is not enough. So you also need to put it to the loss. Apart from the loss that you had before, you will have one more term for the loss. It will be called coverage loss and the idea is to minimize the minimum of the attention probabilities and the coverage vector. Take a moment to understand that. So imagine you want to attend some moment that has been already attended a lot, then this minimum will be high and you will want to minimize it. And that's why you will have to have small attention probability at this moment. On the opposite, if you have some moment with low coverage value, then you are safe to try to have high attention weight here because the minimum will be still the low coverage value, so the loss will not be high. So this loss motivates you to attend those places that haven't been attended a lot yet. Let us see whether the model works nice and whether the coverage trick helps us to avoid repetitions. We can compute the ratio of duplicates in our produced outcomes, and also we can compute the same ratio for human reference summaries, and you can see that it is okay to duplicate unigrams, but it is not okay to duplicate sentences because the green level there is really low, it is zero. So the model before coverage, the red one, didn't know that and it duplicated a lot of three-grams and four-grams and sentences. The blue one doesn't duplicate that, and this is really nice. However, we have another problem here. The summary becomes really extractive, which means that we do not generate new sentences, we just extract them from our input. Again, we can try to compare what we have with reference summaries. Let us compute the ratio of those n-grams that are novel. And you can see that for the reference summaries, you have rather high bars for all of them. So, the model with coverage mechanism has sufficiently lower levels than the model without the coverage mechanism. So in this case, our coverage spoils a model a little bit. And again for the real example, this is the summary generated by pointer-generator network plus coverage, and actually let us see. Somebody says he plans to something. And here in the original text, we see exactly the same sentences but they are somehow linked. So, we just link them with he says that and so on. Otherwise, it is just extractive model that extracts these three important sentences. Now, I want to show you quantitative comparison of different approaches. ROUGE score is an automatic measure for summarization. You can think about it as something as BLEU, but for summarization instead of machine translation. Now, you can see that pointer-generator networks perform better than vanilla seq2seq plus attention, and coverage mechanism improves the system even more. However, all those models are not that good if we compare them to some baselines. One very competitive baseline would be just to take first three sentences over the text. But it is very simple and extractive baseline, so there is no idea how to improve it. I mean, this is just something that you get out of this very straightforward approach. On the contrary, for those models for attention and coverage, there are some ideas how to improve them even more, so in future everybody hopes that neural systems will be able to improve on that, and it is absolutely obvious that in a few years, we will be able to beat those baselines.</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/HyI98/summarization-and-simplification)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


5. Week 5

    11. Natural Language Understanding (NLU)

        * [Task-oriented dialog systems](https://www.coursera.org/learn/language-processing/lecture/05t9K/task-oriented-dialog-systems)

<table>
  <tr>
    <td>Hi. This week, we will talk about task-oriented dialog systems. And where you can see task-oriented dialog systems, you can actually talk to a personal assistant like Apple Siri or Google Assistant or Microsoft Cortana or Amazon Alexa. You can solve these tasks like set up a reminder or find a photos of your pet or find a good restaurant or anything else. So, people are really familiar with this personal assistance and this week we will overview how you can make your own. Okay. You can also write to chat bot like for different reasons: to book a tickets, to order food, or to contest a parking ticket for example. And this time, you don't use your voice but you rather type in your question to the bot and you actually assume that the result will come up instantaneously. What we actually get from the user when he uses our system is either speech or text. If it is speech, we can run it through automatic speech recognition and get the text and the result. And what we actually get is the utterance and we will further assume that our utterance is text and we don't mess with speech or anything like that because it is out of scope of this week. The first thing you need to do when you get the utterance from the user, is you need to understand what does the user want, and this is the intent classification problem. You should think of it as the following, which predefined scenario is the user trying to execute? Let's look at this Siri example, "How long to drive to the nearest Starbucks?", I asked Siri and the Siri tells me the result. The traffic to Starbucks is about average so it should take approximately ten minutes. And I had such an intent, I wanted to know how long to drive to the nearest Starbucks and we can mark it up as the intent: navigation.time.closest. So, that means that I am interested about time of navigation to the closest thing. And I can actually ask it in any other way and because our natural language has a lot of options for that. But it will still need to understand that this is the same intent. Okay. So, I can actually ask the Siri a different question, "Give me directions to nearest Starbucks". This time, I don't care about how long it takes, I just need the directions. And so this time, Siri gives me the directions of a map. And let's say that this is a different intent like navigation.directions.closest. And you actually need to classify different intents, you need to distinguish between them, and this is classification task and you can measure accuracy here. And one more example, "Give me directions to Starbucks." This time, I don't say that I need the time or the nearest Starbucks, that's why the system doesn't know which Starbucks I want. And that's when this system initiate the dialogue with me and because it needs additional information like which Starbucks. And this is intent: navigation.directions. And how to think about this dialogue and how our chat bot, a personal assistant actually tracks what we are saying to it. You should think of intent as actually a form that a user needs to fill in. Each intent has a set of fields or so-called slots that must be filled in to execute the user request. Let's look at the example intent like navigation.directions. So that the system can build the directions for us, it needs to know where we want to go and from where we want to go. So, let's say we have two slots here like FROM and TO, and the FROM slot is actually optional because it can default to current geolocation of the user. And TO slot is required, we cannot build directions for you if you don't say where you want to go. And we need a slot tagger to extract slots from the user utterance. Whenever we get the utterance from the user, we need to know what slots are there and what intent is there. And let's look at slot filling example. The user says, "Show me the way to History Museum." And what we expect from our slot tagger is to highlight that History Museum part, and tell us that History Museum is actually a value of a TO slot in our form. And you should think of it as a sequence tagging and let me remind you that we solve sequence tagging tasks using BIO Scheme coding. And in here B corresponds to the word of the beginning of the slot, I corresponds to the word inside the slot, and O corresponds to all other words that are outside of slots. And if we look at this example, "Show me the way to History Museum.", the text that we want to produce for each token are actually the following, "Show me the way to" are outside of any slots, that's why they have O, "History" is actually the beginning of slot TO, and "Museum" is the inside token in the slot TO, so that's why it gets that tag. You train it as a sequence tagging task in BIO scheme and we have overview that in sequence to sequence in previous week. Let's say that a slot is considered to be correct if it's range and type are correct. And then, we can actually calculate the following metrics: we can calculate the recall of our slot tagging, we can take all the two slots and find out which of them are actually correctly found by our system, and that's how we define a recall. The precision is the following would take all of found slots and we find out which of them are correctly classified slots. And you can actually evaluate your slot tagger with F1 measure, which is a harmonic mean of precision and recall that we have defined. Okay. So, let's see how form filling dialog manager can work in a single turn scenario. That means that we give single utterance to the system and then outputs the result right away. Okay, the user says "Give me directions to San Francisco." We run our intent classifier and it says, "This is in navigation.directions intent." Okay, then we're on slot tagger and it says that "San Francisco seems to be the value of slot TO." Then, our dialog manager actually needs to decide what to do with that information. It seems that all slots are filled so we can actually ask for the route. We can query Google Maps or any other service that will give us the route, and we can output it to the user and say, "Here is your route." Okay, that was a simple way, this is a single dialog. Let's look at a more difficult example. This time the user starts the conversation like this, "Give me directions from L.A.", and we run intent classifier, it says, "Navigation.directions", where on slot tagger and it says that Los Angeles is actually a FROM slot and this time, dialog manager looks at this and says, "Okay, so required slot is missing, I don't know where to go. Please ask the user where to go." And the system asks the user, "Where do you want to go?", and the user gives us, this is where a second turn in the dialog happens and the user says San Francisco. We're on our intent classifier and slot tagger and hopefully, they will give us the values on the slide. The slot tagger will feel that San Francisco barred as TO slot. This time dialog manager knows that, "Okay. I have all the information I need. I can query Google Maps and give you the route." And the assistant outputs, "Here is your route." The problem here is that during the second turn here, actually, if we don't know the history of our conversation and just see the odds are in San Francisco, it's really hard to guess that this is in navigation.directions intent and that San Francisco actually fills TO slot. So, here we need to add context to our intent classifier and slot tagger and that context is actually some information about what happened previously. Let's see how you can track context in an easy way. We already understand that both intent classifier and slot tagger are needed. Let's add simple features to both of them. The first feature is the previous utterance intent as a categorical feature. So we know what to user wanted in the previous turn and that information can be valuable to decide what to do now, what intent the user has now. Then, we also add the slots that are filled in so far with binary feature for each possible slot, so that the system during slot tagging already knows which slots are filled by the user previously and which are not, and that will help it to decide which slot is correct in the utterance it sees. And this simple procedure actually improves slot tagger F1 by 0.5% and it reduces intent classifier error by 6.7%. So, this is pretty cool. These are pretty easy features and you can reduce your error. We will review a better way to do that and that is memory networks but that will happen later. Okay. But how do we track a form switch? Imagine that at first the user says, "Give me directions from L.A.", and then we ask, "Where do you want to go?" and this time, the user says, "Forget about it, let's eat some sushi first." So, this is where we need to understand that the intent has changed and we should forget about all the previous slots that we had and all the previous information that we had because we don't need it anymore. And the intent classifier gives us navigation find and the category, which is a slot and it has the value of sushi. Then, we make a query to the database or knowledge base like Yelp and dialog manager understands, "Okay, let's start a new form and find some sushi." and the assistant outputs, "Okay, here are nearby sushi places." We can actually track the forms which when the intent switches from navigation.directions lets say to navigation.find. If we overview the whole system, it looks like the following: we have a user, we get the speech or text from him or her, and then, we have natural language understanding module that outputs us intents and slots for our utterance. Then we have that magic box that is called dialog manager and dialog manager is responsible for two tasks. The first one is dialog state tracking. So we need to understand what the user wanted throughout the conversation and track that state. And also, it does dialog policy managing. So, there is a certain policy, which says that, okay, if the state is the following then we need to query some information from the user or request some information from the user or we just inform the user about something. And we can also query backend services like Google Maps or Yelp, and when we are ready to give users some information, we use natural language generation box that outputs the speech for the user so that this is a conversation. Okay, so let's summarize. We have overviewed the task-oriented dialog system with form filling and, how do we evaluate form filling? We evaluate accuracy for intent classifier and F1-measure for slot tagger. In the next video, we will take a closer look at the intent classifier and slot tagger. Downloads Lecture Videomp4 Subtitles (English) WebVTT Transcript (English) txt Slides pdf Would you like to help us translate the transcript and subtitles into additional languages?</td>
    <td></td>
  </tr>
</table>


        * [Intent classifier and slot tagger (NLU)](https://www.coursera.org/learn/language-processing/lecture/RmVnE/intent-classifier-and-slot-tagger-nlu)

<table>
  <tr>
    <td>Hi. In this video, we will talk about intent classifier and slot tagger in depth. Let's start with intent classifier. How we can do that. You can use any model on bag-of-words with n-grams and TF-IDF, just use classical approaches of text mining, or you can use some recurrent architecture and you can use LSTM cells, GRU cells, or any other. You can also use convolutional networks and you can use 1D convolutions that we have overviewed in week one. And the study actually shows that CNNs can perform better on datasets where the task is essentially a key phrase recognition task and it can happen in some sentiment detection datasets, for example. So, it makes sense to try RNN or CNN, or any classical approach as a baseline and choose what works best. Then, there comes a slot tagger, and this is a bit more difficult task. It can use handcrafted rules like regular expressions, so that when I say, for example, take me to Starbucks, then you know that if something happens after the phrase take me to, then that is most definitely like a two slot or any other slots of your intent. But that approach doesn't scale because the natural language has a huge variation in how we can express the same thing. So, it makes sense to do something data driven here. You can use conditional random fields, that is a rather classical approach, or you can use RNN sequence-to-sequence model, when you have encoder and decoder, and a funny fact is that you can still use convolutional networks for a sequence-to-sequence task as well, and you can add attention to any of these models, any sequence-to-sequence model. In the next slide, I want to overview convolutional sequence-to-sequence model because that is- that gains popularity because it works faster and sometimes it even beats RNN in some tasks. Okay, let's see how convolutional networks can be used to model sequences. Let's say we have an input sequence which is bedding-bedding, then start of sequence and three German watts. And what we actually want to do, let's say, where we want to solve the task of language modeling. When we see each new token, we need to predict which token comes next. And usually, we use a recurrent architectures for this. But let's see how we can use convolutions. Let's say that when we generate the next token, what we actually- we actually care only about the last three tokens in the sequence that we have seen. And if we assume that, then we can use convolution to aggregate the information about the last three tokens and this is the blue triangle here, and we actually get some filters in the output. Let's take half of those filters and add them as is, and the second half, we will pass through sigmoid activation function, and then take an element Y as multiplication of these two halves. What we actually get is we get some Gated Linear Unit, and we add non-linear part to it and it becomes non-linear. So, this is how we actually look at the context that we had before and we predict some hidden state or let's say, next token and you can use convolutions for that, and then, that triangle is actually convolutional filter and you can slide it across the sequence and use the same weights, the same learned filters, and it will work the same on every iteration on that sequence. So, it is pretty similar to RNN, but in this way, we actually don't have a hidden state that we need to change. We actually only look at the context that we had before, and some intermediate representation. But you can see that we actually look at only three last tokens and that is not very good. Maybe we need to look at it like last 10 tokens or so because RNN is like LSTM cell, can actually have a very long short-term memory. Okay. So, we know from convolutional neural networks, we know how to increase the input receptive field. And we actually stack convolutional layers. Let's stack six layers here with kernel size five, and that will actually result in an input field of 25 elements. And the experiments show that 25 elements in the receptive field might be enough to model your sequences. Let's see how CNNs work for sequences. The office provided the results on language modeling dataset which is WikiText-103, and you can see that this CNN architecture actually beats LSTM, it has lower perplexity, and it actually runs faster. We will go into that a little bit later. And another example is a machine translation dataset, or from English to French, let's say, and there they have a metric called BLEU and the higher that metric the better. And you can see that convolutional sequence-to-sequence actually beats LSTM here as well, and this is pretty surprising. What is a good thing about CNNs is, the speed benefit. If you compare it with RNN, the problem with RNN is that it has a hidden state and we change that state through iterations and we cannot do our calculations in parallel, because every step depends on the other, and we can actually overcome that with convolutional networks because during training, we can process all time steps in parallel. So, we apply the same convolutional filters but we do that at each time step, and they are independent and we can do that in parallel. During testing, let's say, in sequence-to-sequence manner, our encoder can actually do the same because there is no that dependence on the previous outputs and we use only our input tokens, and we can apply that convolutions and get our hidden states in parallel. During testing one more thing, one more good thing is that GPUs are highly optimized for convolutions and we can get a higher throughput, thanks to using convolutions instead of RNNs. You can actually see a table here, and it shows the model based on LSTM, and the model based on convolutional sequence-to-sequence, and you can see that convolutional model actually provides a better score in terms of translation quality, and it also works 10 times faster. So, that is a pretty good thing because for a real-world systems like, let's say Facebook, they need to translate to the post when you want and they need to translate it fast. So, in order to implement these machine translation in production environment, maybe CNN is a very good choice. By the way, this paper is by the folks from Facebook. So, let's look at one more thing. You know that when you do a sequence-to-sequence task, you actually want your encoder to be bi-directional, so that you look at the sequence from left to right and from right to left. And the good thing about convolutions is that actually you can make that convolutional filters symmetric, and you can look at your context at the left and at the right to the same time. So, it is very easy to make bi-directional encoder with CNNs. And it still works in parallel, there is no dependence on hidden state here, it just applies all of that multiplications in parallel. To move further, with our, let me remind you, we are actually reviewing intent classifier and slot tagger and to move further, we need some dataset so that we can use it for our overview. Let's take ATIS dataset, it's Airline Travel Information System. It was collected back in 90s, and it has roughly 5,000 context independent utterances, and that is important. That means that we actually have a one turn dialogue and we don't need like a fancy dialogue manager here. It has 17 intents and 127 slot labels, like from location to location, departure time, and so forth. The utterances are like this, show me flights from Seattle to San Diego tomorrow. The State-of-the-art for this task is the following: 1.7 intent error, and 95.9 slots F1. So, this is pretty cool. Another thing is that you can actually learn your intent classifier and slot tagger jointly. You don't need to train like two separate tasks, you can train this supertask, because it can actually learn representations that is suitable for both tasks, and this time, we provide more supervision for our training and we get the higher quality as a result. Let's see how this joint model might work. It is still a sequence-to-sequence model, but this time we use, let's say, a bi-directional encoder, and the last hidden state, we can use for decoding the slot tags, and at the same time we can use that to decode the intent. And if we train these end-to-end for the two tasks, we can get a higher quality. And notice that we have in the decoder, we have hidden states from encoder post just as is, and this is called aligned inputs, and we also have C-vectors which are attention. Let's see how attention works in decoder. Lets say that we have at time step E, and we have to output our new decoder hidden state SE. And that is actually a function of the previous hidden state which is in blue, a previous output which is in red, and hidden stated from encoder and some vector which is attention. Let's see how attention works. The vector attention Ci, is actually a weighted sum of hidden vectors from encoder. And we need to come up with weights for these vectors. And we actually train the system to learn these weights in such a way so that it makes sense to give attention to those weights, to those vectors. And the coefficient that we use to define what weight that particular vector from encoder has, is modeled as a forward network that uses our previous decoder hidden state, and all of the states from encoders, and it needs to figure out whether we need that state from encoder or not. You can also see an example of attention distribution when we predict the label for the last word, and you can see that when we predict the label like departure time, our model looks at phrases like, from city, or city name, or something like that. Okay. So, we can also see how our two losses decrease during training, and during training we use two losses and we use a sum of them, and you can see the green loss here is for intent, and the blue one is for slots. You can see that intent loss actually saturates and it doesn't change, but blue slots, blue curve continues to decrease and so, our model continues to train because that is a harder task than intent classification. Okay. Let's look at joint training results on the 80s dataset. If we had trained slot filling independently, we have slot F1 95.7, and if we train our intent detection, our classifier independently we have intent at two percent, but if we train those two tasks jointly using the architecture that we have overviewed, we actually can get a higher slot F1 and a lower intent error. And a good thing also is that this joint model works faster if you use it on mobile phone, or any other embedded system because you have only one encoder and you reuse that information for two tasks. Okay. Let's summarize what we have overviewed. We have viewed at different options for intent classifier and slot tagger, you can start from classical approaches and go all the way to deep approaches. People start to use CNNs for a sequence modeling and sometimes get better results than with RNN. This is a pretty surprising fact. You can also use joint training and it can be beneficial in terms of speed and performance for your slot tagger and intent classifier. In the next video, we will take a look at context utilization in our NLU, our intent classifier and slot tagger.</td>
    <td></td>
  </tr>
</table>


        * [Adding context to NLU](https://www.coursera.org/learn/language-processing/lecture/3KFLa/adding-context-to-nlu)

<table>
  <tr>
    <td>Hi. In this video, we'll talk about context utilization in our NLU. Let me remind you why we need context. We can have a dialect like this. User says, "Give me directions from LA," and we understand that we need, we have a missing slot so we ask, "Where do you want to go?" And then the user says, "San Francisco." And when we have the next utterance, it would be very nice if intent classifier and slot tagger could use the previous context, and it could understand that, that San Francisco is actually, @To slot that we are waiting, and the intent didn't change, and we had context for that. A proper way to do this is called memory networks. Let's see how it might work. We have a history of utterances, and let's call them x's, and that is our utterances. Then we passed them through a special RNN, that will encode them into memory vectors. And we take out with two utterances passed through these RNN, and we have some memory vectors. And these are dense vectors just like neural networks like. Okay. So we can encode all the utterances we had before into the memory. Let's see how we can use that memory. Then when a new utterance comes, and this is utterance C in the lower left corner, then we actually encoded into the vector of the same size as our memory, and we use a special RNN for that, called RNN for input. And when we have that, orange "u" vector, we actually, this is actually the representation of our current utterance, and what we need to do is we need to match this current utterance with all the utterances that we had before in that memory. And for that, we use a dark product with the representations of utterances we had before, and that actually gives us, after applying soft marks, we can actually have a knowledge attention distribution. So we know what knowledge, what previous knowledge is relevant to our current utterance and which is not. And we can actually take all the memory vectors, and we can take them with weights of this attention distribution, and we have a final vector which is a weighted sum. We can edit to our representation of an utterance, which is an orange vector, and we can pass it through some fully connected layers and get the final vector "o" which is the knowledge encoding of our current utterance and the knowledge that we had before. What do we do with that vector? That vector actually accumulates all the context of the dialect that we had before. And so, we can actually use it in our RNN for tagging, let's say. Now, let's say how we can implement that knowledge vector into tagging RNN. We can edit as input on every step of our RNN tagger, and that is a memory vector that doesn't change, and if we train it end to end, then we might have a better quality because we use context here. Okay. So this is an overview of the whole architecture. We have historical utterances, and we use a special RNN to turn them into memory vectors. Then we use attention mechanism when a new utterance comes, and we actually know which prior knowledge is relevant to us at the current stage and which is not. And we use that information in the RNN tagger that gives us slot tagging sequence. Let's see how it actually works. If we evaluate the slot tagger on multi-turn data set, when the dialect is along, and we actually measure F1, F1-measure here. And let's compare RNN tagger without context, and these memory networks architecture. We can see that this model performs better and not only on the first turn but also on the consecutive turns as well. And overall, it gives a significant improvement to the F1 score, like 47, comparing with 6 to 7. So, let me summarize. You can make your NLU context-aware with memory networks. In the previous weeks, in the previous videos, we actually overviewed how you can do that in a simple manner, but memory network seems to be the right approach to this. In the next video, we will take a look at lexicon utilization in our NLU. You can think of lexicon as, let's say, a list of all music artists. We already know that this is a knowledge base, and let's try to use that in our intent classifier and slot tagger. </td>
    <td></td>
  </tr>
</table>


        * [Adding lexicon to NLU](https://www.coursera.org/learn/language-processing/lecture/UjdWx/adding-lexicon-to-nlu)

<table>
  <tr>
    <td>In this video, we will talk about lexicon utilization in our NLU. Why do we want to utilize lexicon? Let's take ATIS dataset for example. The problem with these dataset is that it has a finite set of cities in training. And, the thing we don't know is whether the model will work for a new city during testing. And, the good fact is that we have a list of all cities like from Wikipedia or any other source, and we can actually use it somehow to help on a model to detect new cities. Another example, imagine you need to fill a slot like "music artist" and we have all music artists in the database, like musicbrainz.org and you can actually download it, parse it, and use for your NLU. But how can we use it? Let's add lexicon features to our input words. We will overview an approach from the paper, you can see the lower left corner. Let's match every n-gram of input text against entries in our lexicon. Let's take n-grams "Take me," "me to," "san," and "San Francisco," and all the possible ones. And let's match them with the lexicon, with the dictionary that we have for, let's say, cities. And we will say that the match is successful when the n-gram matches either prefix or postfix of an entry from the dictionary, and it is at least half the length of the entry, so that we don't have a lot of spurious matches. Let's see the matches we might have. San might have a match with San Antonio, with San Francisco, and the San Francisco n-gram can match with San Francisco entry. So, we'd get these matches and we need to decide which one of them is best. And when we have overlapping matches, that means that one word can be used in different n-grams, we need to decide which one is better, and we will prefer them in the following order. First, will prefer exact matches over partial. So, if the word San is used in San Francisco and that is an exact match, then it is preferable than, let's say, the match of San with San Antonio. And we will also prefer longer matches over shorter, and we will prefer earlier matches in the sequence over later. This three rules actually give us a unique distribution of our words in the non-overlapping matches with our lexicon. So, let's see how we can use that information, that lexicon matching information in our model. We will use a so-called BIOES coding, which stands for Begin, Inside, Outside, End, and Single, and we will mark the token with B if token matches the beginning of some entity. We will use B and I if token matches as prefix. We will use I and E if two tokens match as postfix. So, it is some token in the middle and some token at the end of the entity. And we will use S for matches when a single token matches an entity. Let's see an example of such coding for four lexicon dictionaries, location, miscellaneous, organization, and person. And we have a certain utterance like "Hayao Tada commander of the Japanese North China area army." And you can see that we have a match in persons lexicon and that gives us a B and E, so we know that that is an entity. And we also have a full match in "North China area army," and it has a match with organisation lexicon, and it has an encoding like B, I, E, I, E. And, we can actually have the full match even if we don't have an entity in our lexicon. Let's say, we have North China History Museum, and let's say, I don't know, any country area army entities. And when we have those two entities, we can actually have the postfix from the second one and the prefix from the first match and it will still give us the same BIOES encoding. So, this is pretty cool. We can make new entities that we haven't seen before. Okay, so, what we do next is we use these letters as we will later encode them as one hot encoded vectors. Let's see how we can add that lexicon information to our module. Let's say we have an utterance, "We saw paintings of Picasso," and we have a word embedding for every token. And to that word embedding, we can actually add some lexicon information. And we do it in the following way. Remember the table that we have on the previous slide? Let's take two first words and let's take that column that corresponds to the word, and let's use one hot encoding to decode that BIOES letters into numbers, and we will use that vector and we will concatenate it with the embedding vector for the word, and we will use it as an input for our B directional LSTM, let's say. And this thing will predict tags for our slot tagger. So, this is like a pretty easy approach to embed that lexicon information into your model. Let's see how it works. It was bench-marked on the dataset for a Named Entity Recognition, and you can see that if you add lexicon, it actually improves your Precision, Recall and F1 measure a little bit, like one percent or something like that. So, it seems to work and it seems that it will be helpful to implement these lexicon features for your real world dialogue system. Let's look into some training details. You can sample your lexicon dictionaries sothat your model learns not only the lexicon features but also the context of the words. Let's say, when I say, "Take me to San Francisco," that means that the word that comes after the phrase "take me to" is most likely a two-slot. And we want the model to learn those features as well because in real world, we can see entities that were not in our vocabulary before, and our lexicon features will not work. So, this sampling procedure actually gives you an ability to detect unknown entities during testing. So, this is a pretty cool approach. When you have the lexicon dictionaries, you can also augment your data set because you can replace the slot values by some other values from the same lexicon. Let's say, "Take me to San Francisco," becomes "Take me to Washington," because you can easily replace San Francisco's slot value with Washington because you have the lexicon dictionaries. So, let me summarize. You can add lexicon features to further improve your NLU because that will help you to detect the entities that the user mentions and some unknown and long entities like "South China area army" that can be detected. In the next video, we will take a look at Dialogue Manager.</td>
    <td></td>
  </tr>
</table>


    12. Dialog Manager (DM)

        * [State tracking in DM](https://www.coursera.org/learn/language-processing/lecture/s9zRt/state-tracking-in-dm)

<table>
  <tr>
    <td>Hi. In this video, we will talk about state tracking in dialog manager. Let me remind you that dialog managers are responsible for two tasks. The first one is state tracking and it actually acquires some hand-crafted states. And, what it does is, it can query the external database or knowledge base for some additional information. It actually tracks the evolving state of the dialog and it constructs the state estimation after every utterance from the user. And another part is policy learner, that is the part that takes the state estimation as input and chooses the next best action from the dialog system from the agent. You can think of a dialog as the following. We have dialog turns, the system, and user provide some input and when we get input from the user, we actually get some observations. We hear something from the user and when we hear something from the user, we actually update the state of the dialog and dialog manager is responsible for tracking that state. Because with every new utterance, user can specify more details or change its intent, and that all affects our state. And you can think of state as something describing what the user ultimately wants. And then, when we have the state, we have to do something, we have to react, and we need to learn policy, and that is a part of a dialogue manager as well. And policy is actually a rule. What do we need to say when we have a certain state? So next, we will all review state tracking part of dialog manager, that red border. And for that, we will need to introduce DSTC 2 dataset. It is a dialog state tracking challenge. It was collected in 2013. It is a human computer dialogs about finding a restaurant in Cambridge. It contains 3,000 telephone-based dialogs and people were recruited for this using Amazon Mechanical Turk. So, this collection didn't assume that we need some experts in the field. These are like regular users that use our system. They used several dialog systems like Markov decision process or partially observed Markov decision process for tracking the dialog state and hand-crafted policy or policy learned with reinforcement learning. So, this is a computer part of that dialog collection. The labeling procedure then followed this principles. First, the utterances that they got from user and that was sound, they were transcribed using Amazon Mechanical Turk as well. And then, these transcriptions were annotated by heuristics, some regular expressions. And then, they were checked by the experts and corrected by hand. That's how these dataset came into being. So, how do they define dialog state and this dataset? Dialog state consists of three things. The first one, goals, that is a distribution over the values of each informable slot in our task. The slot is an informable if the user can give it in the utterance as a constraint for our search. Then, the second part of the state is a method, that is a distribution over methods namely by name, by constraints, by alternatives, or finished. So these are the methods that we need to track. And, user can also request some slots from the system. And, this is a part of a dialog state as well. They requested slots that the user needs and that is a probability for each requestable slot that it has been requested by the user and the system should inform it. So, the dataset was marked up in terms of user dialog acts and slots. So, the utterance like what part of town is it, can become the request. So, this is an act that the user makes. And, you can think of it as an intent, and that area slot that is there tells us that the user actually wants to get the area. Then, we can infer the method from act and goals. So if we have informed food which is Chinese, then it is clear that we need to search by constraints. Let's look at the dialog example. The user says, "I'm looking for an expensive restaurant with Venetian food." What we need to understand from this is now our state becomes food= Venetian, price range=expensive, and the method is by constraints and no slots were requested. Then, when the dialog progresses, the user says,"Is there one with Thai food?" And, we actually need to change our state so all the rest is the same, but food is now Thai. And then, when the user is okay with the options that we have provided, it asks, "Can I have the address?" And that means that our state of our dialog is the same, but this time, the requested slots is the address. And so, these three components goals, method, and requested slots are actually our context that we need to track off to every utterance from the user. So, let's look at the results of the competition that was held after the collection of this dataset. The results are the following. If we take the goals, then the best solution had 65 percent correct combinations that means that every slot and every value is guessed correctly and that happened in 65 percent of times. And as for the method, it has 97 percent accuracy and requested slots have 95 percent accuracy as well. So, it looks like slot tagging is still the most hard, the most difficult part. How can you do that state tracking? When you looked at that example, that was pretty clear that after those utterances, it is pretty easy to change the state of our dialog. So maybe, if you train a good NLU, which gives you intents and slots, then you can come up with some hand-crafted rules for dialog state change. If the user like mentions a new slot, you just add it to the state, if it can override the slot or it can start to fill a new form. And, you can actually come up with some rules to track that state, but you can actually do better if you do neural networks. This is an example of an architecture that does the following. It uses the previous system output, which says, "Would you like some Indian food?" Then, it takes the current utterance from the user like, "No, how about Farsi food?" And then, we need to actually parse that system output and user utterance and to come up with a current state of our dialog. And this is done in the following way. First, we embed the context and that is the system output on the previous state. Then, we embed the user utterance and we also embed candidate pairs for the slot and values, like food-Indian, food-Persian, or any other else. Then, we do the following thing. We have a context modelling network that actually takes the information about system output, about candidate pairs, uses some type of gating and uses the user utterance to come up with the idea whether this user utterance effects the context or not. And also, there is the second part which does semantic decoding, so it takes user utterance, the candidate pairs for slot and values, and they decide whether they match or not. And finally, we have a final binary decision making whether these candidate pairs match the user utterance provided the previous system output was the following. So in this way, we actually solve NLU and dialog state tracking simultaneously in a joint model. So, this is pretty cool. Let's see, for example, how one part of that model can actually work and let's look at the utterance representation. We can take our utterance, we can split it into tokens, we can take Word2Vec embeddings, or any other embeddings you like. And then, we apply 1D convolutions that we investigated in week one. And, you can take bigram, trigram, and so forth. And then, you can just sum up those vectors and that's how we get the representation for the utterance. So, that is a small part in our architecture. And we don't have time to overview like all of those parts. Let's go to the results. If we look at how good that network is, you can see that using that neural belief tracker architecture with convolutional neural networks, you can get 73 percent accuracy for goals, and this is pretty huge improvement. And, it actually improves request accuracy as well on our dialog state tracking challenge dataset. We can see that when we solved the task of NLU and dialog state tracking simultaneously, we can actually get better results. Another dataset worth mentioning is Frames dataset. It is pretty recent dataset. It was collected in 2016. It is human-human goal-oriented dataset. It is all about booking flights and hotels. It has 12 participants for 20 days and they have collected 1400 dialogs. And, they were collected in human-human interaction, that means that two humans talk to each other via a Slack chat. One of them was the user and he has a task from the system. Find a vacation between certain dates, between destination, and like the place where you go from, and date not flexible if not available, then end the conversation. So, the user had this task. The wizard which is another user, which has an access to a searchable database with packages and a hotel, and round trip flights, and that user, his task was to provide the help via a chat interface to the user who was searching for something. So, this dataset actually introduces a new task called frame tracking, which extends state tracking to a setting where several states attract simultaneously and users can go back and forth between them and compare results. Like, I simultaneously want to compare the flight from Atlanta to Caprica or let's say from Chicago to New York, and I investigate these two options, and these are different frames, and I can compare them. So, this is a pretty difficult task. How is it annotated? It is annotated with dialog act, slot types, slot values, and one more thing, references to other frames for each utterance. And also, we have an idea of the current active frame for each utterance. Let's see how it might work. The user says, "2.5 stars will do." What he does is, he actually informs the system that the category equally 2.5 is okay. Then, the system might ask the user. It might make an offer to him, like offer the user in the frame six business suite for the price $1,000 and it will actually be converted into the following utterance from the system. What about a 1,000 business class ticket to San Francisco? And we know that it is to San Francisco because we have an ID of the frame, so we have all the information for that frame. Let's summarize, we have overviewed a state tracker of a dialog manager. We have discussed the datasets for dialog manager training and those are dialog state tracking challenge and Frames dataset. State tracking can be done by hand having a good NLU or you can do better with neural network approaches, like a joint NLU and dialog manager. In the next video, we will talk about dialog policies in dialog managers.</td>
    <td></td>
  </tr>
</table>


        * [Policy optimisation in DM](https://www.coursera.org/learn/language-processing/lecture/HlhWS/policy-optimisation-in-dm)

<table>
  <tr>
    <td>Hi. In this video, we will talk about Policy Learner in Dialogue Manager. Okay, let me remind you what policy learning is. We have a dialogue that progresses with time, and after every turn, after every observation from the user will somehow update our state of the dialogue and state records responsible for that. And then, after we have a certain state, we actually have to make some action, and we need to figure out the policy that tells us if you have a certain state then this is an action that you must do, and this is something that we then sell to the user. So let's look at what dialog policy actually is. It is actually a mapping from dialog state to agent act. Imagine that we have a conversation with the user. We collect some information from him or her, and we have that internal state that tells us what the user essentially wants, and we need to take some action to continue the dialog. And we need that mapping from dialog state to agent act, and this is what dialog policy essentially is. Let's look at some policy execution examples. A system might inform the user that the location is 780 Market Street. The user will hear it as of the following, "The nearest one is at 780 Market Street." Another example is that the system might request location of the user. And the user will see it as, "What is the delivery address?" And we have to train a model to give us an act from a dialog state or we can do that by hand crafted rules, which is my favorite. Okay, so let's look at the Simple approach: hand crafted rules. You have NLU and state tracker. And you can come up with hand crafted rules for policy. Because if you have a state tracker, you have a state, and if you remember the dialog state tracking challenge dataset, it actually contains a part of the state which has requested slots, and we can use that information to understand what to do next, whether we need to tell the user a value of a particular slot or we should search the database or something else. So, it should be pretty easy to come up with hand crafted rules for policy. But it turns out that you can make it better if you do it with machine learning. And there are two ways to do that, to optimize dialog policies with machine learning. The first one is Supervised learning, and in this setting, you train to imitate the observed actions of an expert. So we have some human-human interactions, one of them is an expert, and you just use that observations and try to imitate the action of an expert. It often requires a large amount of expert label data and as you know it is pretty expensive to collect that data, because you cannot use crowd sourcing platforms like Amazon Mechanical Turk. But even with a large amount of training data, parts of the dialog state space may not be well covered in the training data and our system will be blind there. So, there is a different approach to this called Reinforcement learning, and this is a huge field and it is out of our scope, but it is like an honorable mention. Given only rewards signal, now, the agent can optimize a dialog policy through interaction with users. Reinforcement learning can require many samples from an environment, making learning from scratch with real user is impractical, we will just waste the time of our experts. That's why there, we need simulated users based on the supervised data for reinforcement learning. So and this is a huge field and it gains popularity in dialog policies optimization. Let's look at how supervised approach might work. Here is an example of another model that does joint NLU and dialog management policy optimization, and you can see what it does. We actually have four utterances that are all utterances that we got from the user so far. We pass each of them through NLU which gives us intents and slot tagging, and we can also take the hidden vector, the hidden representation of that phrase from the NLU and we can use it for a consecutive LSTM that will actually come up with an idea what system action we can actually execute. So, we've got several utterances, NLU results, and then the LSTM reads those utterances in latent space from NLU, and it actually decides what to do next. So this is pretty cool because, here, we don't need dialog state tracking, we don't have state. State here is replaced with a state of the LSTM, so that is some latent variables like 300 of them let's say. So our state becomes not hand crafted, but it becomes a real valued vector. So this is pretty cool. And then we can actually learn a classifier on top of that LSTM, and it will output us the probability of the next system action. Let's see how it actually works. If we look at the results, there are three models that we compare here. The first one is baseline. That is a classical approach to this problem. We have a conditional random field for slot tagging and we have SVM for action classification. As you can see, the frame level accuracies, that means that we need to be accurate about everything in the current frame that we have after every utterance, and you can see that the accuracy for dialog manager is pretty bad here. But for NLU, it's okay. Then, another model is Pipeline-BLSTM, and what it actually does is it does NLU training separately and then that bidirectional LSTM for dialog policy optimization on top of that model. But these models are trained separately. And you can see that the third option is when these two models, NLU and bidirectional LSTM which was in blue in the previous slides, we can actually train them end to end, jointly, and we can increase the dialog manager accuracy by a huge margin and we actually improve NLU as well. So we have seen that effect of joint training before, and it still continues to happen. Okay, so what have we looked at? Dialog policy can be done by hand crafted rules if you have a good NLU and you have a good state tracker. Or it can be done in a supervised way where you can learn it from data and you can learn it jointly with NLU, and this way you will not need state tracker for example. Or you can do the reinforcement learning way, but that is a story for a different course.</td>
    <td></td>
  </tr>
</table>


        * [Final remarks](https://www.coursera.org/learn/language-processing/lecture/zdM5Q/final-remarks)

<table>
  <tr>
    <td>[MUSIC] Hi, in this video I want to overview
what we have done this week. We have overviewed so-called
task-oriented dialog systems. And our dialog system
looks like the following. We get the speech from the user and
we can convert it to text using ASR. Or we can get text like in chat bots. Then comes Natural Language Understanding
that gives us intents and slots from that natural language. Then, there is a magic box called Dialog
Manager, and it actually does two things. It tracks the dialog state and it learns the dialog policy, what should
be done and what the user actually wants. The Dialog Manager can query a backend
like Google Maps or Yelp or any other. And then it cast to say
something to the user. And we need to convert the text
from Dialogue Manager to speech with some Natural Language Generation. The red boxes here are the parts of
the system that we don't overview because it will take a lot of time. And it can actually work
without those systems. It can take the user input as text,
so you will not need ASR. Then you can output your response
to the user as a text as well. So we don't need
Natural Language Generation. And sometimes you don't need Backend
action to solve the user's task. We have overviewed in details Natural
Language Understanding and Dialog Manager. And let me remind you, you can train
slot tagger and intent classifier, which are basically NLU. And you can train them separately or
jointly. And when you do that jointly,
that yields better results. You can train NLU and Dialogue Manager
separately or jointly, and it will give you better results as well. You can use hand-crafted rules sometimes. For example, for
dialog policy over state tracking. But learning from data actually works
better if you have time for that. Let me remind you how we evaluate NLU and
Dialog Manager. For NLU, we use turn-level metrics
like intent accuracy and slots F1. For Dialogue Manager,
there are two kinds of metrics. The first is turn-level metrics. That means that after every
turn in the dialogue, we track let's say,
state accuracy or policy accuracy. And they're are dialog-level
metrics like success rate, whether this dialog solved
the problem of a user or not or what reward we got when we
solved that problem of the user. The reward could be the number of turns
and we want to minimize that turns, so that we solve that task for
the user faster. And here, actually, is the question. We have NLU and Dialogue Manager. And if we train them separately, we want to understand how
the errors of NLU affect the final quality of our Dialog Manager. Here, on the left, on the vertical axis, we have success rate. And on the right, on the same axis, we have average number of
turns in the dialogue. And we have three colors in the legend. The blue one is when we
don't have any NLU errors. The green one is when we have
10% of the errors in NLU and a red one is when we have
20% of errors in our NLU. And you can see what happens. When you have a huge error in NLU, the success rate of your
task actually decreases. And the number of turns needed to solve
that task where there was a success, actually increases. So it takes more time for
the user to solve his task and the chance of solving that task is lower. But NLU actually consists of
intent classifier and slot tagger. So let's see which one is more important. Let's look what happens when we
change the Intent Error Rate. It looks like it doesn't
effect the quality, the success rate of our
dialogue that much. And the dialogues don't
become that much longer. So it looks like intent error is not as important as slot tagging,
and we will see now why. Because when you introduce the same
amount of error in slot tagging, that actually decreases your success
rate of the dialogue dramatically. And it seems that slot tagging
error is actually the main problem of our success rate. So it looks like we need to
concentrate on slot tagger. And that can give you some insight
when you want to train a joint model. When you have a loss for intent and
a loss for slot tagging. You can actually come up with
some weights for them so that the intuition isn't following. It seems like a slot tagging
loss should have a bigger weight because it is more important for
the success of the whole dialogue. Let me summarize, we have overviewed how
test-oriented dialogue system looks like. And we have overviewed in-depth NLU
component and Dialog Manager component. So this is the basic knowledge
that you will need to build your own task-oriented
dialog system. So that's it for this week, I wish you
good luck with your final project. [MUSIC]
</td>
    <td></td>
  </tr>
</table>


        * [Quizz](https://www.coursera.org/learn/language-processing/exam/dfqCL/task-oriented-dialog-systems)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


    13. Final project: StackOverflow assistant

        * [Peer-graded Assignment: StackOverflow Assistant](https://www.coursera.org/learn/language-processing/peer/xbHJG/stackoverflow-assistant)

<table>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>


        * [Papers mentioned in week 5](https://www.coursera.org/learn/language-processing/supplement/oO6R6/papers-mentioned-in-week-5)

<table>
  <tr>
    <td>Here's a list of papers mentioned in week 5 slides.
Video 1:
A. Bhargava, A. Celikyilmaz, D. Hakkani-Tur, and R. Sarikaya. EASY CONTEXTUAL INTENT PREDICTION AND SLOT DETECTION (2013). http://www.cs.toronto.edu/~aditya/publications/contextual.pdf
K. Scheffler and S. Young. Simulation of human-machine dialogues (1999). http://mi.eng.cam.ac.uk/~sjy/papers/scyo99.ps.gz
Video 2:
Wenpeng Yin, Katharina Kann, Mo Yu, Hinrich Schütze. Comparative Study of CNN and RNN for Natural Language Processing (2017). https://arxiv.org/pdf/1702.01923.pdf
Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier. Language Modeling with Gated Convolutional Networks(2017). https://arxiv.org/pdf/1612.08083.pdf
Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin. Convolutional Sequence to Sequence Learning(2017). https://arxiv.org/pdf/1705.03122.pdf
Bing Liu, Ian Lane. Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling (2016). https://www.isca-speech.org/archive/Interspeech_2016/pdfs/1352.PDF
Gokhan Tur, Dilek Hakkani-Tur, Larry Heck. WHAT IS LEFT TO BE UNDERSTOOD IN ATIS? (2010). https://www.microsoft.com/en-us/research/wp-content/uploads/2010/12/SLT10.pdf
Video 3:
Yun-Nung Chen, Dilek Hakkani-Tur, Gokhan Tur, Jianfeng Gao, and Li Deng. End-to-End Memory Networks with Knowledge Carryover for Multi-Turn Spoken Language Understanding (2016). https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/IS16_ContextualSLU.pdf
Video 4:
Jason P.C. Chiu, Eric Nichols. Named Entity Recognition with Bidirectional LSTM-CNNs (2016). https://arxiv.org/pdf/1511.08308v4.pdf
Video 5:
Xiujun Li, Yun-Nung Chen, Lihong Li, Jianfeng Gao, Asli Celikyilmaz. Investigation of Language Understanding Impact for Reinforcement Learning Based Dialogue Systems (2017). https://arxiv.org/pdf/1703.07055.pdf
Matthew Henderson, Blaise Thomson and Jason Williams. Dialog State Tracking Challenge 2 & 3 (2013). http://camdial.org/~mh521/dstc/downloads/handbook.pdf
Nikola Mrksic, Diarmuid O Seaghdha, Tsung-Hsien Wen, Blaise Thomson, Steve Young. Neural Belief Tracker: Data-Driven Dialogue State Tracking (2017). https://arxiv.org/pdf/1606.03777.pdf
Layla El Asri, et al. FRAMES: A CORPUS FOR ADDING MEMORY TOGOAL-ORIENTED DIALOGUE SYSTEMS (2017). https://arxiv.org/pdf/1704.00057.pdf
Video 6:
Xuesong Yang, Yun-Nung Chen, Dilek Hakkani-Tur, Paul Crook, Xiujun Li, Jianfeng Gao, Li Deng. END-TO-END JOINT LEARNING OF NATURAL LANGUAGE UNDERSTANDING AND DIALOGUE MANAGER (2017). https://arxiv.org/pdf/1612.00913.pdf
</td>
    <td></td>
  </tr>
</table>


        * [Keep up to date with NLP research](https://www.coursera.org/learn/language-processing/supplement/GW1DU/keep-up-to-date-with-nlp-research)

<table>
  <tr>
    <td>Congratulations for finishing with the lectures in our course!
NLP is a huge and rapidly emerging area. So to have an up-to-date understanding of its advances one should always keep track of what is going. In these reading material we provide some links for you that give a nice overview of NLP trends as for the end of 2017.
First, it is always a good idea to check out highlights from main conferences. There are nicely summarized trends of ACL-2017: part 1, part 2. Also, some highlights from EMNLP-2017 are available here. Second, it would be a good idea to monitor some blogs, e.g. Sebastian Ruder has nice posts about DL in NLP, optimization trends, word embeddings, and many others.
One of still active topics is Thought Vectors and how one can interpret directions in the hidden space. E.g. you might be interested to check out this post. However, it's getting more clear that compressing all the input into one vector is often not enough and one might make nice things with attention and linguistic information. Some more tips about attention here.
Finally, this is another nice overview of 2017 trends in NLP research - advances in unsupervised machine translation seem especially exciting!
To conclude, we would like to say thank you for taking our course and wish best of luck in your future NLP projects!
</td>
    <td></td>
  </tr>
</table>


    14. Project extension: custom conversational model

        * Practice

# II. Applied Text Mining

1. [Module 1: ](https://www.coursera.org/lecture/python-text-mining/introduction-to-text-mining-y5C24)

1. [Introduction to Text Mining](https://www.coursera.org/lecture/python-text-mining/introduction-to-text-mining-y5C24)

<table>
  <tr>
    <td>Welcome to the course on applied text mining in Python, I'm glad you're here. Today we're going to start with working with text. Text is everywhere, you see them in books and in printed material. You have newspapers, you have Wikipedia and other encyclopedia.You have people talking to each other in online forums, and discussion groups, and so on. You have Facebook and Twitter, that's most text, too.
And this text data is growing really fast. It grows exponentially and continues to grow so. And it's estimated to be about 2.5 Exabytes, that is 2.5 million TB, a day. 
It'll grow to about 40 Zettabytes, according to recent estimates. That is 40 billion TB by 2020. And that is 50 times that of what was just 10 years ago. 
Approximately 80% of all of this data is estimated to be unstructured and free text. 
That includes over 40 million articles in Wikipedia. Over 5 million of them are in English. Actually, just 5 million of them are in English. 4.5 billion Web pages, about 500 million tweets a day. That consists of about 200 billion tweets a year. And over 1.5 trillion queries on Google in a year. 
So when we look at data and look at what is hidden in text in plain sight, you'll see that it says a lot. 
So for example, this is the Twitter profile of UN Spokesperson. So you have the author there. And you have description, location of where they are. You have the tweets, themselves, the actual content, if you think about it. That has the topic and the sentiment around each of them. For each tweet, you have the timestamp, when it was sent out. You have the popularity of how many times this is retweeted or liked by others. And in general, this also gives you some idea of the social network. About how many people are following this account, how many accounts are being followed by this particular account of UN Spokesperson. 
So what can be done with all of this text? 
You could parse the text, try to understand what it says. Find and extract relevant information from text, even define what information is. You're to classify the text documents. You're to search for relevant text documents, this is information retrieval. 
You're to also do some sort of sentiment analysis. You could see whether something is positive or negative. Something is happy, something is sad, something's angry. They're all sentiments associated with a particular piece of text. And then you could do topic modeling, identify what is a topic that is being discussed. How many topics are being discussed in this document and so on? 
We are going to talk about each of these in the next few modules. And see what we can do using text mining in Python. 
</td>
    <td>Chào mừng bạn đến với khóa học về text mining trên Python, tôi rất vui khi bạn ở đây. Hôm nay chúng ta sẽ bắt đầu với làm việc với văn bản. Văn bản ở khắp mọi nơi, bạn thấy chúng trong sách và trong tài liệu in, ví dụ như  báo, Wikipedia và bách khoa toàn thư khác.Những người nói chuyện với nhau trong các diễn đàn trực tuyến, và các nhóm thảo luận, v.v. hay Facebook và Twitter, hầu hết đều là text.
Và dữ liệu văn bản này đang phát triển rất nhanh. Nó phát triển theo cấp số nhân và vẫn đang tiếp tục phát triển. Và ước tính khoảng 2,5 Exabyte, tức là 2,5 triệu TB, một ngày.Nó sẽ tăng lên khoảng 40 Zettabyte, theo ước tính gần đây. Đó là 40 tỷ TB vào năm 2020. Và gấp 50 lần so với 10 năm trước.
Khoảng 80% của tất cả các dữ liệu này được ước tính là không có cấu trúc và văn bản tự do.Bao gồm hơn 40 triệu bài viết trên Wikipedia. Hơn 5 triệu bài viết trong số đó bằng tiếng Anh. Trên thực tế, chỉ có 5 triệu trong số đó là tiếng Anh. 4,5 tỷ trang web, khoảng 500 triệu tweet mỗi ngày. Điều đó bao gồm khoảng 200 tỷ tweets một năm. Và hơn 1,5 nghìn tỷ truy vấn trên Google trong một năm.
Vì vậy, khi chúng ta nhìn vào dữ liệu và nhìn vào những gì ẩn sâu trong văn bản một cách trực diện, bạn sẽ thấy rằng nó nói lên rất nhiều điều.Vì vậy, ví dụ, đây là hồ sơ Twitter của người phát ngôn của LHQ. Vì vậy, bạn biết được tác giả. Và bạn có thông tin mô tả, vị trí của họ. Bạn biết tweet, những gì thuộc về bản thân họ, nội dung thực tế, nếu bạn nghĩ về nó. Điều đó có chủ đề và dư luận xung quanh mỗi người trong số họ. Đối với mỗi tweet, bạn có dấu thời gian, khi nó được gửi đi. Bạn có sự phổ biến của bao nhiêu lần này được retweeted hoặc thích bởi những người khác. Và nói chung, điều này cũng cung cấp cho bạn một số ý tưởng về mạng xã hội. Khoảng bao nhiêu người đang theo dõi tài khoản này, có bao nhiêu tài khoản đang được theo sau bởi tài khoản đặc biệt của Người phát ngôn của LHQ.
Vì vậy, những gì có thể được thực hiện với tất cả các văn bản này?Bạn có thể phân tích cú pháp văn bản, cố gắng hiểu những gì nó nói. Tìm và trích xuất thông tin liên quan từ văn bản, thậm chí xác định thông tin nào là. Bạn đang phân loại tài liệu văn bản. Bạn đang tìm kiếm các tài liệu văn bản có liên quan, đây là thông tin truy xuất.Bạn cũng làm một số phân tích ý kiến dư luận. Bạn có thể thấy một cái gì đó là tích cực hay tiêu cực. Một cái gì đó là vui, một cái gì đó là buồn, một cái gì đó gây phẫn nộ. Tất cả đều là ý kiến về với một đoạn văn bản cụ thể. Và sau đó bạn có thể làm mô hình hóa chủ đề, xác định chủ đề đang được thảo luận là gì. Có bao nhiêu chủ đề đang được thảo luận trong tài liệu này và như vậy?Chúng ta sẽ nói về từng cái trong vài mô-đun tiếp theo. Và xem những gì chúng ta có thể làm bằng cách sử dụng khai thác văn bản bằng Python.

</td>
  </tr>
</table>


2. [Handling Text in Python](https://www.coursera.org/lecture/python-text-mining/handling-text-in-python-MeheH)

<table>
  <tr>
    <td>In this video we're going to talk about handling text in Python. Let's first start looking at primitive constructs in text. You have sentences or strings and they are formed of words or tokens, and words are formed out of characters. On the other side, you have documents and larger files and we're talking about all these constructs and their properties. So let's try it out. Let's pull out a sentence from the U.N. spokesperson's twitter profile and say, that is text1. So text1 here is "Ethics are built right into the ideals and objectives of the United Nations." If you find out the length of text1, it could tell you how many characters are there in this string. That's 76. What if you want to know the words? So you have to split this text on space.  that is our primitive tokenization. So you split this sentence on space to find out words or tokens. And the length of that is 13. There are 13 tokens in the sentence. And what are those? Ethics, are, built, right, in, doing, so on. So these all look very good. They all are valid words. So this is great. Looks like this splitting works. Now, if you are to find specific words for example, long words that are more than three characters long, you would say w for w in text2, where if length of w is greater than three. And that will give you all these words that are more than three characters long in this text2. Ethics, built, right, into and so on. What if we want to find out capitalized words? Capitalized words are those that start with a capital letter A to Z, but you could use something like istitle because istitle is a function that checks whether the first character is capitalized and the others are small. So w for w in text2 if w.istitle will tell you that the w.istitle is true for words like Ethics, United and Nations and it's false otherwise. What if you want to find out words that end with s. You can say w for w in text2 if w.endswith s,` that will give you Ethics, ideas, objectives, nations. Great. So now we have found out how to find individual words. Now let's look at finding unique words and where to use set function for that. Let's take another example text3, that is this famous phrase "To be or not to be." If you split it in space, you are going to get six words- to, be, or, not, to, be six of them. Now if you use set function, it's going to find out all the unique words in this list. So when you say set of text 4, it's going to find out unique words, that will be: to, be, or, not. So we expect four. But we get the answer of 5. What happened? If you look at the set text4, you'll see that you do have to, be, or, not, but you have "to" occuring twice. One with a capital T and other with the small t. That's a problem because you don't want to have these two variants just because one was the first word and was capitalized. So to fix that we should lowercase the text. So we say w.lower for w in text4 and then find set of that and the length of that and that will give you 4. So if you print the entire set, it is indeed to be or not in some order. Great. Now let's look in more detail on some of the word comparison functions. We have startswith, we have endswith as we saw with endswith s. We can also use a function call t in s to find out the substrings. If a particular substring is in a largest string, and then you have these functions that check whether a particular string is capitalized- isupper, is all small case. Lower case islower or title case- where the first one is capitalized and the others are small, using s.istitle. The same way you can check for other patterns. You can check for isalpha which is whether a particular string is alphanumeric. Isdigit, if it is actually, isalpha is if the string is just made of alphabets, is digit if the string is just made of digits 0 to 9 and isalnum is if the string is made of alphabets or numerals or digits- that's isalnum. Once we have done this checking operations, you can look at more on the string operations. We have already seen s.lower where it takes in a string s and gives out the lowercase version of that string. It could use s.upper to make the entire string uppercase, or titlecase to make it title case. You can split a sentence s on a smaller string t. So if you split something on space, , then t becomes that space- one character and we have seen that that will give out words from a sentence. The same way you could use split lines. So s.splitlines is going to split a sentence on the newline character or end of line character- \n in some cases. s.join is the opposite of splitting. So you have s.join on t, would say that you think the words represented by an array or a set in t and join it using a string that is s. You can also do some cleaning operations on string s.strip is going to take out all the whitespace characters, which means spaces and tabs from the front of the string, and rstrip is something that will take out these spaces and whitespace characters and Tabs and so on from the end of the string. Let's take an example. In fact, s.strip is going to take these whitespace characters from the front and the back. s.find is going to find a particular substring t in s, from the front. While s.rfind is going to find the string t from s from the end of the string. Finally, s.replace, it takes two parameters, u and v, where every occurrence of u, a smaller string in s, is going to be replaced by v, and other small string. So let's take these examples and see how it works. So first look at words two characters. text5 is ouagadougou. For those who know, this is the capital city of Burkina Faso. And I like this word in general because of this repetitions of characters. So you split that sentence or word in this case text5 on ou. What do you expect to see? We'll see that in text5, when you split it with ou, you're going to get four groups. The first is an empty string because the string text5 starts with ou. So there is nothing before that. That's what that empty means. And then between the first occurrence of ou and the second occurrence of ou, you have agad. That's the second element in this set. And then you have g at the third and then finally ou is the last set. Last set of characters in ouagadougou, so there is nothing after, so the fourth one is also empty. So when you have a particular string ou occurring three times in text, in this case, text5, when you split it, you're going to get four parts: Before the first, between the first and the second, between the second and the third, and after the third. Okay, So that's why you have this four. And now if you join text6, that is this array of four elements using the string ou, you are going to get back ouagadougou. That's where we started, so split and join are opposites of each other. Now suppose we want to find characters- all the characters in the word. We would imagine that we have to split on something. So let's split on empty. So text5.split on an empty string should give you all the characters. But actually what you get is an error. It says that it's an empty separator. So it doesn't work. So what should we do? The way to do that is to find list of text5. So when you have text5, which is a string, list of text5 gives you all the characters, and then you could also- the other way to do that would be for c in text5. So that will also give you the string with- sorry the array with individual characters. So there are two ways really to get characters out of words. One would be do use list function and the other is to say c for c in text5. Now let's take some examples of cleaning text. So we take an example text8. That is a string, "a quick brown fox jumped over the lazy dog" but it has some whitespace characters before and after. When it's split on space, because there are whitespace characters before and after, you're going to get these empty strings at the start or a tab at the start and so on. Because there are indeed multiple spaces and tabs up in front and there's also a space at the end, so we have this empty string at the end. This is not how you want to get all the words when you have just stray spaces around some text. So, what we'll do is we say text8.strip. Note that strip, strips out whitespace characters both from the start and the end. So then if you split on space, you're going to get the sentence as it is, and you basically get the words right. A quick brown fox and so on. What if you want to change the text, like finding and replacing. So remember text9 was this "a quick brown fox jumped over the lazy dog" and  we want to find the character o. In this string, when you say text9.find o is going to give you the offset, the characters offset of where it found the first o. So in fact, that was at character 10, because a is character one, and then space is character two, and so on. In fact, a would be character zero, because at zero bound, it's 0, 1 and then quick becomes two three four five six, then the space at seven, and eight, nine, and 10. So brown, the o in middle of Brown would be character number 10. The same way you could do rfind that's reverse find, and that will give you the character of 14 because if you start from the first a, as zero and go to the o in dog, that's character number 14. Finally, you can replace And let's replace the small o with a capital O. And that will give you the same sentence, The quick brown fox jumps over the lazy dog. But every occurrence of O that's four of them, has been replaced by capital O. So this demonstrates how you can use find and rfind and replace to change text. What about handling larger texts? Larger texts are typically going to be in files. So you're going to read the files and you have to read them , line by line. And in this case let's take an example of a file that I have called UNHDR.txt. This is the United Nations Declaration of Human Rights. I say that you have to open it using fopen UNDHR.txt and you have to open it and read more. So that is r. Once you have that. And you say f.readline is going to read the first line, and that is, "Universal Declaration of Human Rights." with the \n to tindicate that it's the end of the line. If you are reading the full file, you can either do it in a loop. Read the line one by one. And because we have already opened it and I want to reset the reading point  to the start of the file, I'm going to say f.seek zero. That resets the reading. And then I could just use f.read and f.read is going to read this entire file and give it back as text12. Then if you look at length of text12, it has read the entire files, so that are 10,891 characters there, that would be your length. So it's not just one sentence, one line out of this file. And then you can split lines on text12 to do basically give you a 158 sentences. In fact, they are not sentences, I should say that they are lines that are delimited by a \n, so a newline character. So for all purposes here, when it's a line, you mean something that ends with the \n. So that are 158 such lines. The first line is Universal declaration of human rights. You will notice that when you do split lines and read the line that way, the \n goes missing because you split it on \n, And so when you split on a particular character, that character is not included in the result. Whereas the f.readline up at the top was reading the line, one line at a time. So it has this \n at the end. In general, these are the file operations that you would want to use. You have open that has a file name and a mod, So read more would have an r, write more would have a w, and so on. You can read the line usingf.readline or f.read or f.read with n, that will read n characters rather than the entire file. For you can use a loop, say, for line in f, doSomething line. You can say f.seek that will reset the reading pointer, let's say reading position.  So f.sek of zero will reset the reading position back to the start of the file. f.write would be how you would write a particular message into a file if you opened it in the appropriate more, in the write mode, say. And then once you've done all the operations, the counterpart of opening is closing. So f.close will close that file handle. And you can check whether something has closed with that particular right handle closed by using f.closed. So when you read this you'll notice that f.readline gave you this \n at the end. That is not necessarily something we want to keep. So how would you take that out? How to take that last newline character out? You could do rstrip. Remember rstrip is used to remove whitespace characters and \n is one of them- from the end of the string. So rstrip would just remove that and give you "Universal declaration of human rights. " And this rstrip works for DOS newline characters that show up as \r or \r\n and so on. So it is universally the function that you would want to use rather than saying find \n, because that may not be the character in the kind of encoding you have. So the take home messages here, as we looked at how to handle text sentences. We saw how to split a sentence into words, and words into characters. We saw two ways to do that. We've set and looked at that how you could find unique words and looked briefly into how to handle a text from documents or from large files. Next, we're going to go in more detail about how you could process the text to find out some interesting concepts from within the text. </td>
    <td>Trong video này, chúng ta sẽ nói về xử lý văn bản bằng Python. Đầu tiên chúng ta hãy bắt đầu xem xét các cấu trúc thuần túy trong văn bản. Ta có câu hoặc chuỗi ký tự và chúng được hình thành từ các từ hoặc các ký hiệu, và các từ được hình thành từ các ký tự. Mặt khác, bạn có tài liệu và tệp lớn hơn và chúng ta đang nói về tất cả các cấu trúc và thuộc tính của chúng. Vì vậy, hãy thử xem xét. Hãy trích thử một câu từ hồ sơ twitter của người phát ngôn của người phát ngôn và những lời nói, đó là text1. Vì vậy, text1 ở đây là "Đạo đức được xây dựng ngay vào những lý tưởng và mục tiêu của Liên Hợp Quốc." Nếu bạn tìm hiểu độ dài của văn bản 1, nó có thể cho bạn biết có bao nhiêu ký tự trong chuỗi này. Đó là 76. Nếu bạn muốn biết từ ngữ thì sao? Bạn phải chia văn bản này trên một khoảng trắng. Giả sử đó là mã thông báo thuần túy của chúng ta. Vì vậy, bạn chia câu này trên quy phạm để tìm ra các từ hoặc thẻ. Và chiều dài là 13. Có 13 thẻ trong câu. Và đó là những gì? Đạo đức, được xây dựng, đúng, trong, làm, vân vân. Tất cả đều là những từ hợp lệ. Có vẻ như việc chia tách này hoạt động. Bây giờ, nếu bạn muốn tìm các từ cụ thể, ví dụ, các từ dài dài hơn ba ký tự, bạn sẽ nói w cho w trong text2, nếu chiều dài của w lớn hơn ba. Và điều đó sẽ cung cấp cho bạn tất cả những từ này dài hơn 3 ký tự trong văn bản 2 này. Đạo đức, được xây dựng, đúng, vào và vân vân. Điều gì sẽ xảy ra nếu chúng ta muốn tìm ra các từ viết hoa? Các từ được viết hoa là những từ bắt đầu bằng chữ cái viết hoa từ A đến Z, nhưng bạn có thể sử dụng cái gì đó như istitle vì istitle là một hàm kiểm tra xem ký tự đầu tiên có được viết hoa không và các ký tự còn lại nhỏ. Vì vậy, w cho w trong text2 nếu w.istitle sẽ cho bạn biết rằng w.istitle là đúng cho các từ như Ethics, United và Nations và đó là sai trong các trường hợp còn lại. Nếu bạn muốn tìm ra những từ kết thúc bằng s. Bạn có thể nói w cho w trong text2 nếu w.endswith s, tìm kiếm sẽ trả về cho bạn các từ Đạo đức, ý tưởng, mục tiêu, quốc gia. Vì vậy, bây giờ chúng ta đã tìm ra cách để tìm các từ riêng lẻ. Bây giờ chúng ta hãy nhìn vào việc tìm kiếm các từ duy nhất và nơi để sử dụng chức năng thiết lập cho điều đó. Chúng ta hãy lấy một ví dụ khác là text3, đó là cụm từ nổi tiếng "To be or not be be." Nếu bạn chia tách nó, bạn sẽ nhận được sáu từ to, be, or, not, to, be. Bây giờ nếu bạn sử dụng chức năng thiết lập, nó sẽ tìm ra tất cả các từ duy nhất trong danh sách này. Vì vậy, khi bạn nói bộ văn bản 4, nó sẽ tìm ra các từ độc lập, đó sẽ là: để, được, hoặc, không. Và chúng ta mong chờ kết quả là bốn. Nhưng chúng ta nhận được câu trả lời của 5. Điều gì đã xảy ra? Nếu bạn nhìn vào tập hợp text4, bạn sẽ thấy rằng bạn phải, phải, hoặc, không, nhưng bạn có "để" xảy ra hai lần. Một với một T vốn và khác với t nhỏ. Đó là một vấn đề bởi vì bạn không muốn có hai biến thể này chỉ vì một là từ đầu tiên và được viết hoa. Vì vậy, để khắc phục rằng chúng ta nên viết chữ thường. Vì vậy, chúng ta nói w.lower cho w trong text4 và sau đó tìm thấy bộ đó và chiều dài của điều đó và điều đó sẽ cung cấp cho bạn kết quả là 4.
Vì vậy, nếu bạn in toàn bộ, nó thực sự là có hoặc không theo một thứ tự nào đó. Bây giờ hãy xem chi tiết hơn về một số hàm so sánh từ. Chúng ta đã bắt đầu, chúng ta có endswith như chúng ta đã thấy với endswith s. Chúng ta cũng có thể sử dụng một chức năng gọi t trong s để tìm ra các chất nền. Nếu một chuỗi con cụ thể nằm trong một chuỗi lớn nhất, và sau đó bạn có các hàm này kiểm tra xem một chuỗi cụ thể có viết hoa-isupper hay không, là tất cả các trường hợp nhỏ. Trường hợp viết hoa hoặc viết hoa chữ thường thấp hơn, trong đó chữ hoa đầu tiên được viết hoa và các chữ cái còn lại nhỏ, sử dụng s.istitle. Cùng một cách bạn có thể kiểm tra các mẫu khác. Bạn có thể kiểm tra isalpha là liệu một chuỗi cụ thể có phải là chữ số hay không. Isdigit, nếu nó là thực sự, isalpha là nếu chuỗi chỉ được làm bằng bảng chữ cái, là chữ số nếu chuỗi chỉ được làm bằng chữ số 0-9 và isalnum là nếu chuỗi được làm bằng bảng chữ cái hoặc chữ số hoặc chữ số đó là isalnum. Khi chúng ta đã thực hiện thao tác kiểm tra này, bạn có thể xem thêm về các hoạt động chuỗi. Chúng ta đã thấy s.lower nơi nó lấy một chuỗi và đưa ra phiên bản chữ thường của chuỗi đó. Nó có thể sử dụng s.upper để làm cho toàn bộ chuỗi chữ hoa, hoặc titlecase để làm cho nó là trường hợp tiêu đề. Bạn có thể chia một câu trên một chuỗi nhỏ hơn t. Vì vậy, nếu bạn chia một cái gì đó trên quy phạm, , sau đó t trở thành quy phạm đó - một nhân vật và chúng ta đã thấy rằng điều đó sẽ đưa ra các từ trong một câu. Cùng một cách bạn có thể sử dụng các cách ngắt dòng. Vì vậy, s.splitlines sẽ chia một câu trên ký tự dòng mới hoặc ký tự kết thúc của dòng— \ n trong một số trường hợp. s.join là ngược lại của việc chia tách. Vì vậy, bạn có s.join trên t, sẽ nói rằng bạn nghĩ rằng các từ đại diện bởi một mảng hoặc một tập hợp trong t và tham gia nó bằng cách sử dụng một chuỗi đó là s. Bạn cũng có thể thực hiện một số thao tác làm mịn trên chuỗi s.strip sẽ lấy tất cả các ký tự khoảng trắng, có nghĩa là dấu cách và các tab từ phía trước của chuỗi, và rstrip là thứ sẽ loại bỏ các khoảng trắng và các ký tự trắng và các thẻ và như vậy từ cuối chuỗi. Hãy lấy một ví dụ. Trên thực tế, s.strip sẽ lấy các ký tự khoảng trắng này từ phía trước và mặt sau. s.find sẽ tìm một chuỗi con t cụ thể trong s, từ phía trước. Trong khi s.rfind sẽ tìm chuỗi t từ s từ cuối chuỗi. Cuối cùng, s.replace, nó lấy hai tham số, u và v, trong đó mỗi lần xuất hiện của u, một chuỗi nhỏ hơn trong s, sẽ được thay thế bởi v và chuỗi nhỏ khác. Vì vậy, hãy lấy những ví dụ này và xem nó hoạt động như thế nào. Vì vậy, đầu tiên nhìn vào từ hai ký tự. text5 là ouagadougou. Đối với những người biết, đây là thủ đô của Burkina Faso. Và tôi thích từ này nói chung vì sự lặp lại của các nhân vật. Vì vậy, bạn chia câu hoặc từ đó trong trường hợp này là text5 trên ou. Bạn mong đợi điều gì? Chúng ta sẽ thấy rằng trong text5, khi bạn chia nó với ou, bạn sẽ nhận được bốn nhóm. Đầu tiên là một chuỗi rỗng vì chuỗi text5 bắt đầu bằng ou. Vì vậy, không có gì trước đó. Đó là những gì có nghĩa là trống rỗng. Và sau đó giữa sự xuất hiện đầu tiên của ou và sự xuất hiện thứ hai của ou, bạn có agad. Đó là yếu tố thứ hai trong tập này. Và sau đó bạn có g ở vị trí thứ ba và cuối cùng là ou là tập cuối cùng. Tập hợp cuối cùng của nhân vật trong ouagadougou, vì vậy không có gì sau, vì vậy thứ tư cũng là sản phẩm nào. Vì vậy, khi bạn có một chuỗi cụ thể ou xảy ra ba lần trong đoạn văn, trong trường hợp này, text5, khi bạn chia nó, bạn sẽ nhận được bốn phần: Trước tiên, giữa đầu tiên và thứ hai, giữa thứ hai và thứ ba, và sau lần thứ ba.
Được rồi, Vì vậy, đó là lý do tại sao bạn có bốn điều này. Và bây giờ nếu bạn tham gia text6, đó là mảng của bốn yếu tố bằng cách sử dụng chuỗi ou, bạn sẽ lấy lại ouagadougou. Đó là nơi chúng tôi bắt đầu, vì vậy chia tay và tham gia là đối lập của nhau. Bây giờ giả sử chúng ta muốn tìm các ký tự - tất cả các ký tự trong từ. Chúng ta sẽ tưởng tượng rằng chúng ta phải chia ra điều gì đó. Vì vậy, hãy chia tay trống. Vì vậy, text5.split trên một chuỗi rỗng sẽ cung cấp cho bạn tất cả các ký tự. Nhưng thực tế những gì bạn nhận được là một lỗi. Nó nói rằng đó là một dấu phân cách trống. Vì vậy, nó không hoạt động. Vậy chúng ta nên làm gì? Cách để làm điều đó là tìm danh sách text5. Vì vậy, khi bạn có text5, đó là một chuỗi, danh sách của text5 cung cấp cho bạn tất cả các ký tự, và sau đó bạn cũng có thể-cách khác để làm điều đó sẽ được cho c trong text5. Vì vậy, điều đó cũng sẽ cung cấp cho bạn chuỗi - xin lỗi mảng với các ký tự riêng lẻ. Vì vậy, có hai cách thực sự để có được nhân vật trong số các từ. Một sẽ được sử dụng chức năng danh sách và khác là nói c cho c trong text5. Bây giờ chúng ta hãy lấy một số ví dụ về làm sạch văn bản. Vì vậy, chúng tôi lấy một ví dụ text8. Đó là một chuỗi, "một con cáo màu nâu nhanh chóng nhảy qua con chó lười" nhưng nó có một số ký tự khoảng trắng trước và sau. Khi nó được phân chia trên khoảng trắng, bởi vì có các ký tự khoảng trống trước và sau, bạn sẽ nhận được các chuỗi trống này ở đầu hoặc một tab ở đầu và cứ như vậy. Bởi vì thực sự có nhiều khoảng trắng và các tab ở phía trước và cũng có một khoảng trắng ở cuối, vì vậy chúng tôi có chuỗi trống này ở cuối. Đây không phải là cách bạn muốn có được tất cả các từ khi bạn có chỉ là khoảng trắng đi lạc xung quanh một số văn bản. Vì vậy, những gì chúng tôi sẽ làm là chúng tôi nói text8.strip. Lưu ý rằng dải, dải ra các ký tự khoảng trắng cả từ đầu và cuối. Vì vậy, sau đó nếu bạn chia nhỏ khoảng trắng, bạn sẽ nhận được câu như nó là, và bạn về cơ bản có được những lời đúng. Một con cáo màu nâu nhanh và vân vân. Nếu bạn muốn thay đổi văn bản, như tìm và thay thế. Vì vậy, hãy nhớ text9 là "một con cáo màu nâu nhanh chóng nhảy qua con chó lười" và giả sử chúng ta muốn tìm nhân vật o. Trong chuỗi này, khi bạn nói text9.find o sẽ cung cấp cho bạn độ lệch, các ký tự được bù đắp ở nơi nó tìm thấy o đầu tiên. Vì vậy, trên thực tế, đó là nhân vật 10, bởi vì một nhân vật là một, và sau đó khoảng trắng là nhân vật hai, và như vậy. Trong thực tế, một sẽ là ký tự số không, bởi vì tại không ràng buộc, nó là 0, 1 và sau đó nhanh chóng trở thành hai ba bốn năm sáu, sau đó khoảng trắng ở bảy, và tám, chín, và 10. Vì vậy, màu nâu, o ở giữa Brown sẽ là nhân vật số 10. Cùng một cách bạn có thể làm rfind đó là tìm kiếm ngược, và điều đó sẽ cung cấp cho bạn nhân vật của 14 bởi vì nếu bạn bắt đầu từ đầu tiên a, bằng không và đi đến o trong chó, đó là ký tự số 14. Cuối cùng, bạn có thể thay thế hãy thay thế o nhỏ bằng vốn O. Và điều đó sẽ cho bạn cùng một câu, Con cáo màu nâu nhanh chóng nhảy qua con chó lười. Nhưng mọi sự xuất hiện của O là bốn trong số chúng, đã được thay thế bằng vốn O. Vì vậy, điều này chứng tỏ làm thế nào bạn có thể sử dụng tìm và rfind và thay thế để thay đổi văn bản. Điều gì về xử lý văn bản lớn hơn? Các văn bản lớn hơn thường có trong các tệp. Vì vậy, bạn sẽ đọc các tập tin và bạn phải đọc chúng ta , từng dòng. Và trong trường hợp này, chúng ta hãy lấy một ví dụ về một tệp mà tôi đã gọi là UNHDR.txt. Đây là Tuyên ngôn Nhân quyền của Liên Hiệp Quốc. Tôi nói rằng bạn phải mở nó bằng cách sử dụng fopen UNDHR.txt và bạn phải mở nó và đọc thêm. Vì vậy, đó là r. Một khi bạn có điều đó. Và bạn nói f.readline sẽ đọc dòng đầu tiên, và đó là, "Tuyên bố chung về Nhân quyền." với \ n để xác định rằng đó là kết thúc của dòng. Nếu bạn đang đọc tập tin đầy đủ, bạn có thể làm điều đó trong một vòng lặp. Đọc từng dòng một. Và bởi vì chúng tôi đã mở nó và tôi muốn thiết lập lại điểm đọc,  đến phần đầu của tập tin, tôi sẽ nói f.seek zero. Điều đó đặt lại việc đọc. Và sau đó tôi chỉ có thể sử dụng f.read và f.read sẽ đọc toàn bộ tập tin này và đưa nó trở lại dạng text12. Sau đó, nếu bạn nhìn vào chiều dài của văn bản12, nó đã đọc toàn bộ các tệp, vì vậy có 10,891 ký tự ở đó, đó sẽ là độ dài của bạn. Vì vậy, nó không chỉ là một câu, một dòng trong tập tin này. Và sau đó bạn có thể chia các dòng trên text12 để làm cơ bản cho bạn 158 câu. Trong thực tế, chúng không phải là câu, tôi nên nói rằng chúng là những dòng được giới hạn bởi một \ n, do đó, một ký tự dòng mới. Vì vậy, cho tất cả các mục đích ở đây, khi đó là một dòng, bạn có nghĩa là một cái gì đó kết thúc bằng \ n. Vì vậy, đó là 158 dòng như vậy. Dòng đầu tiên là tuyên bố phổ quát về nhân quyền. Bạn sẽ nhận thấy rằng khi bạn chia các dòng và đọc dòng theo cách đó, \ n bị thiếu vì bạn chia nó thành \ n, Và vì vậy khi bạn chia nhỏ một ký tự cụ thể, ký tự đó không được bao gồm trong kết quả. Trong khi f.readline ở trên cùng là đọc dòng, một dòng tại một thời điểm. Vì vậy, nó có \ n này ở cuối. Nói chung, đây là các thao tác tệp mà bạn muốn sử dụng. Bạn đã mở mà có một tên tập tin và một mod, Vì vậy, đọc thêm sẽ có một r, viết nhiều hơn sẽ có một w, và như vậy. Bạn có thể đọc dòng usingf.readline hoặc f.read hoặc f.read với n, sẽ đọc n ký tự chứ không phải toàn bộ tệp. Ví dụ, bạn có thể sử dụng một vòng lặp cho dòng f, doSomething. Bạn có thể nói f.seek sẽ đặt lại con trỏ đọc, giả sử vị trí đọc. Vì vậy, f.sek của số không sẽ thiết lập lại vị trí đọc trở lại phần đầu của tập tin. f.write sẽ là cách bạn sẽ viết một tin nhắn cụ thể vào một tập tin nếu bạn mở nó trong thích hợp hơn, trong chế độ ghi, nói. Và sau đó một khi bạn đã thực hiện tất cả các hoạt động, các đối tác của việc mở cửa là đóng cửa. Vì vậy, f.close sẽ đóng tập tin đó. Và bạn có thể kiểm tra xem một cái gì đó đã đóng với quyền xử lý cụ thể đóng bằng cách sử dụng f.closed. Vì vậy, khi bạn đọc này, bạn sẽ nhận thấy rằng f.readline đã cho bạn \ n này ở cuối. Đó không nhất thiết phải là thứ chúng ta muốn giữ. Vì vậy, làm thế nào bạn sẽ đưa nó ra? Làm cách nào để lấy ký tự dòng mới nhất đó ra? Bạn có thể làm rstrip. Hãy nhớ rstrip được sử dụng để loại bỏ các ký tự khoảng trắng và \ n là một trong số chúng - từ cuối chuỗi. Vì vậy, rstrip sẽ chỉ loại bỏ điều đó và cung cấp cho bạn "Tuyên bố chung về nhân quyền." Và rstrip này hoạt động cho các ký tự dòng mới của DOS hiển thị dưới dạng \ r hoặc \ r \ n và cứ tiếp tục như vậy. Vì vậy, nó là phổ quát các chức năng mà bạn sẽ muốn sử dụng hơn là nói tìm \ n, bởi vì đó có thể không phải là nhân vật trong các loại mã hóa bạn có. Vì vậy, đưa tin nhắn về nhà ở đây, khi chúng tôi xem xét cách xử lý các câu văn bản. Chúng tôi đã thấy cách chia một câu thành các từ và các từ thành các ký tự. Chúng tôi đã thấy hai cách để làm điều đó. Chúng tôi đã thiết lập và xem xét cách bạn có thể tìm thấy các từ duy nhất và xem xét ngắn gọn về cách xử lý văn bản từ tài liệu hoặc từ các tệp lớn. Tiếp theo, chúng ta sẽ đi vào chi tiết hơn về cách bạn có thể xử lý văn bản để tìm ra một số khái niệm thú vị từ bên trong văn bản.





</td>
  </tr>
</table>


3. [Regular Expressions](https://www.coursera.org/lecture/python-text-mining/regular-expressions-sVe8B)

<table>
  <tr>
    <td>
In this video, we are going to talk about regular expression. When you're processing free-text, we come across a lot of places where regular expressions or patterns play a role. 

So, let us take an example. 

This is one of the tweets from the UN spokesperson's account and this is ethics are built right into the ideals and objectives of the United Nations. We have seen this earlier today and then you have #UNSG. There is another little piece of text like @ New York society for ethical culture. There's a URL there and then two callouts to @UN and @UN_Women. So from this tweet, if you were find out all callouts and hashtags, how would you do it? First, of course, you have to split it on space to get these individual tokens. So, you have that. So now, you know that you have all of these individual tokens. And now, this is in a better position to find out where callouts are or hashtags are. Why don't you give it a try? 

So when you're finding out specific words, such as callouts or hashtags, you may want to see what patterns these hold. So for example, hashtags start with the hash symbol or the pound sign. So, you can use something like this. You can say, w for w in text11 if w.startswith the hash symbol. Great, you get #UNSG. That's exactly what you want, great. 

What about callouts? Same with hashtags, you can say, callouts are strings that start with the symbol at three, the @ symbol. So you can say, w for w in text11 if W starts with @ and you'll get the response as @UN_Women. But there is also this @ separately. Recall that there was this @newyorksociety and that @ as a separate token comes in. That's not really a callout. 

So, this doesn't really work. So now, what should we do? It's not sufficient to say that something starts with the @. The callouts are tokens that begin with @. They have to follow that up with something. So for example, these are valid callouts @UN_Spokesperson or @katyperry or @coursera. So what you can see is that it's something that has to match after @ like alphabets or numbers, or special symbols like the underscore sign. 


So what you're saying is the pattern is @ and then capital letters or small letters and number and underscore occurring one, or more times. 

That is the pattern that defines callouts. 

So, let’s try this one out. So you have the text, text10 and then you can split it on space and then you do just check for first character to be @. So when you sa, text for all words in text11 that start with @, you get @ and @UN and @UN_Women. But if you first use a regular expression and import re, that is import a regular expression package and say, all words where words in text11 if re.search. That means if you can search and find this pattern @ and then A to Za. A to Za-z0-9_ occurring multiple times in the word, then that is a valid token and that would give you the two callouts from that tweet. So, let's look at this callout regular expression in more detail. We said, @ and then we had this square bracket. That's at A to Za to z0 to 9_. And then after the square bracket, we had a +. What does that mean? So it says that you have to start with an @ and then just to follow one or more times, that's what the + means. One or more times. That's at least once, but may be more of something that is in between the square brackets. And that the one that was in between the square brackets is any alphabet, uppercase or lowercase or a digit or underscore and any suspicious symbol that is allowed in a callout expression. So in this regular expression, you saw that we used some notation and meta-characters that we're going to define now. 

So in regular expressions, a dot is a wildcard character that matches a single character. 

Any character, but just once. 

When we use the character symbol, that indicates the start of the string. So even though not all strings start explicitly the character sign, this character matches an empty character at the start of the string. 

The counterpart of that is dollar symbol that represents the end of the string. So if your string has a backslash end, the dollar comes after the backslash end. So, there's a backslash end and a dollar. Dollar is truly the end of the string, the way it represented. The square brackets matched one of the characters that are within the square bracket. 

You have a to z, for example, within square bracket that rematch one of the range of characters a, b up to z. There are some changes there when you have square brackets, but caret abc. It means something different. It means it matches a character that is not a or b, or c. So, it's kind of the inverse of it. So when I said that opening bracket and closing square bracket indicates that it matches any other characters within it, this particular expression with a character the star doesn't mean that it matches a character, an a or a b or a c. But it says, explicitly that it is. It matches any other character, but a, b or c. 

A or b would match either a or b where a and b are themselves strings. You can use this normal braces to say, scoping for operators. You'll use this backslash as escape character, especially for special characters like \t and \n and you could use special character symbols. Just like \t is for tab, you have \b to match a word boundary. You have \d to match any digit, any single digit 0 to 9. That's included into square bracket 0 to 9. \d is any non-digit, anything that is not 0 to 9. So, that's the equal interpretation there, square brackets not 0 to 9. \s is any whitespace character. That is matches space or a tab, or a new line, or \r and \f and \v. Capital S. So, \S is matches any non-whitespace character just the way d and capital D are opposites of each other. S and capital S are opposite of each other. Same way you have w. \w matches any alphanumeric character whereas \W matches any non-alphanumeric character, then we look at repetitions. When you have star, that says that anything that occurs before that matches zero or more times. So, it may not happen at all or it can happen multiple times. Once, twice, thrice or so on. If it's plus, it has to happen once. At least once, but it can happen more than once too. If it's a question mark, it means that it has to match once or does not match at all. So, it matches zero times or once. And now, you have these curly braces. So curly braces and n, it means it matches exactly n times where n is more than 0. If you have curly braces three, it means that this has to match thrice and only thrice. If it's n comma, it means it's at least intense. And if it's comma n, it means it's most intense. 

If it's a combination of both of them m comma n, it means that it has to match at least m times and at most n times. Let's look back at the callout expression. When we use this callout expression using this re.search, we do get @UN, @UN_Women. But then if you say for w when re.search that \w actually matches A-Za-z0-10_, so that would also give you the same answer, any alpha numeric symbol character. Let's look at some more examples. If you were to find some special characters, for example, if you have the text Ouagadougou. Again, the capital city of Burkina Faso. And in order to find all vowels, you'll say, re.findall and the regular expression in square bracket is a, e, i, o, u and that will give you all the vowels that are there. 


Same way, if you are to find out everything that is not a vowel. That's a consonant, then you could use the carrot a, e, i, o, u option and that will give you g, d and g. These are the only three consonants in Ouagadougou. 

So now, let's take a look at a special case to find regular expressions for dates. 

Dates are unique in the sense that they have multiple variants in free-text. Let's take a date off the top of my head. Let's say, 23rd October 2002 and look at variants of how you could write this date. You could write as date, month and the year. 

Either with a hyphen or the slash with two digits for the year other than four. If you are in the US, then you would use month, date, year format. You might use words for October rather than the number ten. So, 23 Oct 2002 or 23 October. And again, change the order where you have month first before date. So, you can see that there are so many variants and top on to other right a regular expression to match it, and write something like this where you have two digits with a slash or a dash, and then four digits that are just going to match some of them. Suppose you have all of these variants, and you use just the first four. You could use this regular expression of two digits followed by a slash and a dash, two digits, slash and a dash then four. And that will give you three of them, because it doesn't match the one that said, yours could be two digits. So then, you have to fix that. So you say, it could be two digits or four digits and then it will match all four. 

And then you could say, the dates it so happen in this its October. So, it's 10. What about September and it would be nine? You have to kind of address that as well or third of October. That will be just one single digit for that. So in general, you have to have one or two characters, sorry, digits for the date. One or two digits for the month and two, or four digits for the year. That will give you all of them here, but that's more children then what you had up here. Now let's look at the other variance where you have October written out, spell it out. Then, you have to say that you have to have two digits of a start, a space and then Jan, Feb, March, April as in J-A-N and F-E-B and so on. One of them occurring, so you use this byte symbol to indicate disjunctions between them and then you have four digits at the end for the year. That will match just October. What happened? Shouldn't it match the whole thing? 

It did match the whole with the date and so on. 

Well, the difference here is that this bracket sign has two meanings. In regular expressions, when you use the bracket, it also indicates scoping. So it says, I want to pull out only something that matched that part. It doesn't match the whole string, but it pulls out and gives you back only the thing that matched between Jan, Feb, March up to December. That's what gave me October as O-C-T. 

For you to find and give it a test and you say, this is a scoping operating operator, but don't pull out just what you matched here. You use this question mark colon special character sequence to indicate that. So when you use that, you'd get the entire string 23 October 2002. Great, why didn't others match? Because the next string was Octobers, spell out completely. We didn't have that and the other two had a comma there, and we didn't have that. So if we were to somehow say that we are going to use the same string, but then we can exchanged Oct to October by saying, it's starts with these three characters, but then it could have a to z multiple times. Why not a plus there? Because May is there with just May. There's nothing that follows may. So you'll have that will match both of them between Oct, 2002 and 23 October 2002. And then finally, we have the two remaining cases where you say that I don't necessarily have to have data to start. I might have data at the end. So the first occurrence of date is you have a question mark after that and then you add a date option at the end, and then a year after that. So you have a question mark, colon, \d(2), comma. Calculating both at the start and at the end before you have the year \d(4), originally. That will match all four. You see that when you're building the regular expressions, you are to keep working on them to finally get the pattern that you'll really want. Note that all of these did not handle this third October case or 23rd of September case, so you have to do all of that with one or two Digits for the date and the month. So, that has to change. And in fact, you may also want to do a \d(2) comma 4 for the year if that is a pattern that you want to fix. Great. So, what did we learn? In this video, we learned about regular expressions. We saw what they are. Why they are useful. What are the regular expression meta-characters and how do you build a regular expression to identify dates. But in general, to identify any particular pattern string that you want to identify. 

In next videos, we are going to see more information about how do you handle other variance of text that you're using. 
</td>
    <td></td>
  </tr>
</table>


4. [Demonstration: Regex with Pandas and Named Groups](https://www.coursera.org/lecture/python-text-mining/demonstration-regex-with-pandas-and-named-groups-wh4nJ)

<table>
  <tr>
    <td>
Hi everyone. Today we'll be looking at working with text data and pandas. 

First let's take a look at our data frame. 

It currently only has the one column, text, and each entry is a string containing a day of the week and a time. 

Using the str attribute we are able to access the set of string processing methods to make it easy to operate on each element in the series. 

For example, applying str.len to the text column shows the number of characters for each string in the series. 

Similarly, we could use str.split to split each string on white space, then use str.len to find the number of tokens for each element of the series. 

To check if a string contains a pattern, we can use str.contains. 

Here, we can see which entries contain the word appointment. 

Using str.count, we can count occurences of a pattern in each string of the series. 

Here, we are finding how many times a digit occurs in each string. We can find out what these occurences are using str.findall. We can also use regular expression capturing groups to group patterns of interest. 

Suppose we wanted to pull out the hour and minutes from each string. 

By capturing these different groups with parenthesis, str.findall is able to return the groups to us. 

Next, let's take a look at how to use str.replace on our series. 

Let's replace any instances of a week day with three question marks. 

To do this we pass an irregular expression that finds the words that end in day and we pass in the three question marks. 

Suppose you wanted to use str.replace to make a change that was based on the original. This is possible using str.replace and lambda expressions. 

Let's take a look at an example where we want to take the weekday from each string and replace it with a three letter abbreviation. 

First, we use the expression that finds the weekday for us and we can create a group by placing it in parenthesis. 

Then for the replacement parameter, we can pass in a lambda expression. 

In this case, we use groups to get a tuple of the groups, index 0 to get the first, and in this case, only group, and slice the first three letters from the group. 

Looking at the results, we can see Monday has been replaced with Mon, Tuesday with Tue, and so on. 

The next method we're going to look at is str.extract, which allows us to quickly create new columns, using the extracted groups. 

For example, if we use extract with the hour and minutes pattern, from the find all example, we get a dataframe with two columns, one for the hours and one for the minutes. 

Note that str.extract only extracts groups from the first match of the pattern. 

To get all matches use str.extractall. 

Let's try an example with extractall that uses a more complex pattern with more groups. 

We'll add a capturing group for am or pm following the hours and minutes, and add parentheses to the entire regular expression, to capture the whole pattern as another group. 

Looking at the data frame returned, we have a multi index where the level match, indicates the order of the match. 

Here we can see there are two matches from the last entry of df text. The columns correspond to the capturing groups with the whole pattern returns in column 0, the hours in column 1, the minutes in column 2, and the period in column 3. 

One last thing I'd like to show you is name groups in regular expressions. 

We can name a group by adding ?P and the name and angle brackets after the first parentheses of the group. 

Extractall will then use the group names as the column names of the returned data frame. 

Here we've added the name time to the first group, hour to the second group, minute to the third group, and period to the fourth group. 

Pandas provides many additional methods for text data, so be sure to check the Pandas working with text data documentation. 

By using and combining these methods you'll be able to do some very powerful text processing of your own with Pandas. Thanks for watching. Hope to see you again next time. 
</td>
    <td></td>
  </tr>
</table>


5. [Internationalization and Issues with Non-ASCII Characters](https://www.coursera.org/lecture/python-text-mining/internationalization-and-issues-with-non-ascii-characters-V7XBv)

<table>
  <tr>
    <td>
In this video we are going to talk about internationalization. The world is a mix of multiple language. We talk about English almost all the time, but English is going more and more as a minority language over the Internet. 

English is very well encoded using ASCII. ASCII stands for American Standard Code for Information Interchange. And this was the encoding scheme that captures all characters. It's 7-bit long, so you have 128 valid characters or valid codes that you could use. And in hexadecimal form it ranges from 0x00 to 0x7F. So if you think about it it takes the seven bits out of eight bits and uses this lower half of the eight bit encoding. This model and this encoding scheme has been in used for quite sometime in start of computing really. It includes alphabets both uppercase and lowercase, it has all ten digits, all punctuations. All common symbols like brackets for example or percentage sign and the dollar symbol, and the hash symbol. It has some control characters, some characters to describe let's say end of the line or the tab or some other control characters that are needed to see. A paragraph ends, for example, is different from a line end. 

And it has worked relatively well for English typewriting. 

I say relatively because even in English it doesn't really capture everything. Let's think about diacritics marks, resume versus resume. If you ignore the diacritics they are the same word, they have the same six characters R-E-S-U-M-E. But the e is different in resume and that is not ready encoded in the ASCII schema things. And this is not the only one. Now e with an i and an umlaut on top of it is another example of that. So is cafe or something with a diacritic on e. It's fairly common in other languages though. And as we have seen that English is not necessarily the majority language anymore, you have to be more sensitive to those changes as well. So for example the names of cities, Quebec or Zurich. Or the names of organizations, like the Federation Internationale De Football Association, or FIFA, which is the leading Football Association. 

But then you have other languages that have completely different encoding scheme, completely different characters, like Chinese, or Hindi, or Greek, or Russian. But then you have languages of music. And you have the musical symbols that are also relevant, especially if you're using the digital form to now create music. And you have to somehow encode the music symbols. And then you have the very famous emoticons. You have the smiley faces and in fact now, many, many more emoticons that are available. 

And all of these need to be included in some way. So what used to be enough for English, let's say, barely enough for English, is definitely not enough when you're including diacritic marks and the multitude of languages that are available. 

In fact there are quite a few different written scripts that need inquiry. 

For example Latin which is basically English is 36% of data and over 2.6 billion people. 36% of people, sorry, use Latin-based languages. And when I say Latin-based languages, I'm including Italian in it because it's a same type of language as French, and so on, if you kind of ignore the diacritics. But then there are completely different written scripts in Chinese. 

Or Devanagari that is the basis for a lot of Indian languages. Or Arabic or Cyrillic and Dravidian which is another Indian language in the south of India. 

So this map kind of shows you how different scripts are used across the world. 

You have a lot of character encodings that have developed now to somehow encode them in computing schemes. So you have the IBM EBCDIC, which is an 8-bit encoding. You have Latin-1 encoding, just slightly different from the ASCII encoding. You have individual country specific standards like JIS for Japanese Industrial Standards and the CCCII that's 

the Chinese Character Code for Information Interchanges like ASCII. 

You have the Extended Unix Code, EUC and there are numerous other national standards. 

But then there was a need to standardize it and bring it all together and that's what Unicode and UTF-8 encoding does. As you'll see there have been a lot of interest in converting the pages to UTF-8. The ASCII only encoding has significantly dropped and consistently dropped over the last 10, 12 years. While UTF-8 was not really that popular until 2004 and 2006. And then just shot up, and is by far the most common encoding used on webpages currently. 

Well, so what is Unicode? Unicode is an industry standard for encoding and representing text. It has over 128,000 characters from 130 odd scripts and symbol sets. So when I'm saying symbol sets, I include Greek symbols, for example, or symbols for the four suits in a card deck and so on. 

It can be implemented using different character endings but UTF-8 is one of them and, by far, the most common one. UTF-8 is an extendable encoding set. It goes from one byte up to four bytes You have UTF-16 which uses one or two 16-bit codes. The UTF-8 is 8 bit sets, so 8 bits is a byte, so you have 1 byte as a minimum and then it goes up to 4 bytes, while UTF 16 is a 16 bit sets one or two of them. UTF-32 is one 32 bit encoding, so even though you have all of these kind of using up to 32 bits, UTF-8 kind of uses one big 32-bit encoding for all characters. 

So what is UTF-8? UTF-8 stands for Unicode Transformational Format- 8-bits, to distinguish it from 16 bit and 32 bit UTF formats. You have variable length encoding, tt goes from one byte to four bytes. And it is also backward compatible with ASCII. So for example, all ASCII codes that were seven bit codes and used the leading zero 

in ASCII are the same codes in UTF-8. So all those that where encoded using ASCII use one byte of information in UTF-8 and uses one byte that is similar to what ASCII says. But then ASCII sort of UTF-8 because it can be extended to four bytes. It has lot of other schemes to kind of do that and keep on extending it. UTF-8 is the dominant character encoding for the Web and in fact, it is 

inherently handled well, it's default in Python 3. If you are one of those who are using Python 2 then you have to give this 

statement in the start of your Python script that tells the interpreter that the coding is, that you are using is UTF-8. So let's see an example. I see the word Resume in both Python 3 and Python 2. So if you enter the same way of text1 as Resume and you find the length of that word. 

In Python 3 you are going to see that it is 6 characters long, 

or I should say it's 6 characters, yeah. 

While in Python 2 you get 8. So why is that? 

Let's see the text itself and you'll see that for Python 3 kind of gives back the word resume. 

But then in Python 2, if you print it out you will see that you have these special hexadecimal codes. So there's a hexadecimal code c389 in place of the diacritic e. Or again you have the c389 at the end of the word. So you would know that these two symbols together the c389 is a representation of diacritic e. And it is represented as two separate bytes in Python 2. So specifically if you want all of these characters you're going to see that it is the six characters come out very well in Python 3. While in Python 2 you're going to get them as eight separate characters. So c3 is one character and a9 is one because it's one byte each. 

So how do you do it? How do you handle UTF-8 strings in Python 2? You're going to use the u before you write your resume. And that would then give the length as 6. It'll tell the interpreter in a way that you have this other Unicode string, a UTF-8 string. And if you actually dubbed individual characters you're going to see that you get R. But then you have the xe9 and then you have these characters again. That hexadecimal called E9 is the Unicode equivalent of the diacritic e. So the take home concept here is that there is a lot of diversity in text, especially text that you will see. And hence it is important for us to recognize that and for computer systems and encodings to capture that. 

ASCII and other character encodings who were used extensively earlier and because there was no standardization, the UTF-8 encoding kind of became popular and is used almost exclusively now. And it's the most popular in coding set. We saw that in Python 3, it is handled by default, so you don't have to do anything specific. While for Python 2 you may have to do something to let the interpreter know that you're using the UTF-8 encoding. 
</td>
    <td></td>
  </tr>
</table>


2. Module 2:

1. [Basic Natural Language Processing](https://www.coursera.org/lecture/python-text-mining/basic-natural-language-processing-AZCCB)

<table>
  <tr>
    <td>Welcome back. This module, we are going to talk about Basic Natural Language Processing, and how it relates to the Text Mining in Python that we have been talking about. So, what is Natural Language? Well, any language that is used in everyday communication by humans is natural language. As compared to something that is artificial language or a computer language like Python. Languages such as English, or Chinese, or Hindi, or Russian, or Spanish are all natural languages. But you know, also the language we use in short text messages or on tweets is also, by this definition natural language, isn't it? 

So then we have to kind of address these as well. 

So then, what is Natural Language Processing? Any computation or manipulation of natural language to get some insights about how words mean and how sentences are constructed is natural language processing. 

One thing to consider when we look at natural language is that these evolve. For example, new words get added. Like selfie or photobomb. 

Old words lose popularity. 

How often have you used thou shalt? 

Meanings of words change. 

Words such as learn in Old English meant exactly opposite of what it means now. It used to mean teach. 

And then language rules themselves may change. So for example, in Old English the position of the verb was at the end of the sentence, rather than the middle as we come to know today. 

So when we talk about NLP tasks, what do we mean? 

.... It could mean as simple as counting the words or counting the frequency of a word or finding unique words in a corpus, and then build on to find sentence boundaries or parts of speech to tag a sentence with its part of speech, parse the sentence structure, try to understand more grammatical constructs and see whether they apply for a particular sentence. 

Identify semantic roles of how these words play. For example, if you have a sentence like Mary loves John. Then you know that Mary is the subject. John is the object, and love is the verb that connects them. 

You have the other NLP tasks like identifying entities in a sentence. So this is called name entity recognition, and in our previous example of Mary loves John, Mary and John are the two entities. Both persons in that sentence. 

And then you could have more complicated, more complex tasks like finding which pronoun refers to which entity. This is called co-ref resolution, or co-reference resolution. 

And there are many, many more tasks that you would do for on free text. 

The challenge is how to do that in an efficient manner, and how it applies to overall text mining. And you're going to see some of those in the next few videos. </td>
    <td></td>
  </tr>
</table>


2. [Basic NLP tasks with NLTK](https://www.coursera.org/lecture/python-text-mining/basic-nlp-tasks-with-nltk-KD8uN)

<table>
  <tr>
    <td>In this video, we are going to talk about basic NLP tasks and introduce you to NLTK. So what is NLTK? NLTK stands for Natural Language Toolkit. It is an open source library in Python, and we're going to use it extensively in this video and the next. The advantage of NLTK is that it has support for most NLP tasks and also provides access to numerous text corpora. So let's set it up. We first get NLTK in using the import statement, you have import NLTK and then we can download the text corpora using nltk.download. It's going to take a little while, but then once it comes back you can issue a command like this from nltk.book import * and then it's going to show you the corpora that it has downloaded and made available. You can see that there are nine text corpora. Text1 stands for Moby Dick, text2 is Sense and Sensibility, you have a Wall Street Journal corpus in text7, you have some Personals in text8 and Chat Corpus in text5. So it is quite diverse here. So as I said text1 is Moby Dick. If you look at sentences, it will show you one sentence each from these nine text corpora and sentence one, Call me Ishmael is from text1. And then, if you look at how sentence one looks, sent1 and you see that it has four words. Call me Ishmael and then full stop. Now that we have access to text corpus and multiple text corpora, we can look at counting the vocabulary of words. So text7 if you recall was Wall Street Journal and sent7 which was one sentence from text7 is this, [ 'Pierre', 'Vinken', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.' ] You have already these words passed out. So you have comma, separate, and you have full stop separate. So the length of sent7 is the number of tokens in this sentence and that's 18. But if you look at length of text7 that's the entire text corpus, you'll see that Wall Street Journal has 100,676 words. It's clear that not all of these are unique. We can see in the previous example that comma is repeated twice and full stop is there and words such as "the" and "a" and so on, are so frequent that they are going to take up a bunch of words from this 100,000 count. So if you see the unique number of words using the command length of set of text7 you'll get 12,408. That means that Wall Street Journal corpus has really only 12,400 unique words even though it is a 100,000-word corpus. Now that we know how to count words, let's look at these words and understand how to get the individual frequencies. So if you want to type out the first 10 words from this set, first 10 unique words, you'll say, list(set(text7))[:10]. That would give you the first 10 words. And in this corpus, the first 10 words really in the set are, 'Mortimer' and 'foul' and 'heights' and 'four' and so on. You can notice that there is this 'u' and a quote before each word. Do you recall what it stands for? You'd recall from the previous videos that 'u' here stands for the UTF-8 encoding. So these have been automatically UTF-8 encoded. So each token is represented as a UTF-8 string. Now, if you have to find out frequency of words, you're going to use this command, frequency distribution, FreqDist and then you create this frequency distribution from text7 that is the Wall Street Journal corpus and store it in this variable called "Dist" you can start finding statistics from this data structure. So you have length of Dist and that will give you 12,408. These are the set of unique words in this word corpus, this Wall Street Journal Corpus. Then, you have dist.keys that gives you the actual words. And that would be your vocab1. And then if you take the first 10 words of vocab1, you will get the same 10 words as we saw up there in the top of the slide. And then, if you want to find out how many times a particular word occurs, you can say, "Give me the distribution of this word four," that is UTF encoded and I'll get the response of 20. That means in this Wall Street Journal corpus, you have four appearing 20 times. What if you want to find out how many times a particular word occurs and also have a condition on the length of the word. So if you have to find out frequent words and say that I would call a word as frequent if that word is at least length five and occurs at least a hundred times, then I can use this command saying w for w in vocab1 if length of w is greater than five and dist of w is greater than 100. And then, I'll get this list of words that satisfy both conditions and you see million and market and president and trading are the words that satisfy this. Why did we have a restriction on length of the word? Because if you don't then words like the or comma or full stop are going to be very very frequent and those will occur more than 100 times and they would come up as frequent words. So this is one way to say, "Oh, you know, the real unique words are ones that are fairly long, at least five characters and occurs fairly often." There are, of course, other ways to do that. Now, if you look at the next task. So we know how to count words, how to find unique words. The next task becomes normalizing and stemming words. What does that mean? Normalization is when you have to transform a word to make it appear the same way or the count even though they look very different. So for example, there might be different forms in which the same word occurs. Let's take this example of input1 that has a word list in different forms. You have it capitalized. You have it plural, lists, you have listings and listings and listed as a verb in the past tense and so on. So the first thing you would want to do is to lowercase them. Why? Because you don't want to distinguish the capital list with small case list. So lower would bring it all to lowercase. And then if you split it on space, you'll get five words, list, listed, lists, listing, and listings. So that was normalization. Then, comes stemming. Stemming is to find the root word or the root form of any given word. You can have multiple algorithms to do stemming. The ones that are quite popular and used widely is Porter stemmer and NLTK gives you access to that. nltk.PorterStemmer would create a stemmer and we call it Porter. And then, if you stem a word using the Porter stemmer, you will get the word list for all of them. So no matter whether it is list or listed or listing, it still gives you the stem of a word as list. This is advantageous because you can now count the frequency of list as the list word occurring itself or in any of its derivation forms, any of its morphological variants. Do you want to do it that way? That's a call that you have to make. You really want to distinguish list and listing, which has slightly different meaning. So you may probably not want to do that but you may want to do list and lists to be merged together and just count as one word. So it is a matter of choice here. Porter stemmer has a particular algorithm to do it and it just makes all of these words the same word, list. A slight variant of stemming is lemmatization. Lemmatization is where you want to have the words that come out to be actually meaningful. Let's take an example. NLTK has a corpus of the universal declaration of human rights as one of its corpus. So if you say nltk.corpus.udhr, that is the Universal Declaration of Human Rights, dot words, and then they are end quoted with English Latin, this will give you all the entire declaration as a variable udhr. So, if you just print out the first 20 words, you'll see that Universal Declaration of Human Rights and there is a preamble and then it starts as whereas recognition of the inherent dignity and of the equal and inalienable rights of people and so on. So it continues that way. Now, if you use the Porter stemmer on these words and get the stemmed version, you'll see that it takes out these common suffixes. So universal became universe without really an e at the end and declaration became declar, and of is of and human right is the same, rights became right, and so on. But now you see that univers and declar are not really valid words. So lemmatization would do that stemming, but really keep the resulting tense to be valid words. It is sometimes useful because you want to somehow normalize it, but normalize it to something that is also meaningful. So we could use something like a wordnet lemmatizer that NLTK provides. So you have nltk.WordNetLemmatizer and then if you lemmatize the word from the set that you've been looking so far, what you get is universal declaration of human rights preamble, whereas recognition of the inherent dignity, so basically all these words are valid. How do you know that lemmatizer has worked? If you look at the first string up there and then the last string down here, rights has changed to right. So it has lemmatized it. But you will also notice that the fifth word here, universal declaration of human rights is not lemmatized because that is with a capital R, it's a different word that was not lemmatized to right. But if you had them in lower case, then the rights would become right again, okay? So there are rules of why something was lemmatized and something was kept as is. Once we have handled stemming and lemmatization, let us take a step back and look at the tokens themselves. The task of tokenizing something. So recall that we looked at how to split a sentence into words and tokens and we said we could just split on space. Right? So if you take a text string like this text11 is, "Children shouldn't drink a sugary drink before bed. " And you split on space, you'll get these words. Children shouldn't as one word, drink a sugary drink before bed, but unfortunately, you have a full stop that goes with bed. So it's bed full stop. Okay. So you got, one, two, three, four, five, six, seven, eight – you got eight words out of this sentence. But you can already see that it is not really doing a good job because, for example, it is keeping full stop with the word. So you could use the NLTK's inherent or inbuilt tokenizer, the way to call it would be nltk.word_tokenize and can pass the string there and you'll get this nice tokenized sentence. And in fact, it differs in two places. Not only is full stop taken away as a separate token, but you will notice that shouldn't became should and this "n't" that stands for "not", and that is important in quite a few NLP task because you want to know negation here. And the way you would do it would be to look for tokens that are a representation of not. So "n't" is one such representation. But now you know that this particular sentence does not really have eight tokens but 10 of them because you've got "n't" and full stop has two new tokens. So we talked about tokenizing a particular sentence and the fact that these punctuation marks have to be separated, there are some unique words like n apostrophe t that should also be separated and so on. But there is even more fundamental question of, what is a sentence and how do you know sentence boundaries? And the reason why that is important is because you want to split sentences from a long text sentence, right? So suppose this example of text12 is, this is the first sentence. A gallon of milk in the U.S. costs $2 99. And is this a third sentence, a question mark. And yes, it is with an exclamation. So already, you know that a sentence can end with a full stop or a question mark or an exclamation mark and so on. But, not all full stops and sentences. So for example, U dot S dot, that stands for US is just one word, has two full stops, but neither of them end the sentence. The same thing with $2.99. That full stop is an indicator of a number but not end of a sentence. We could use NLTK's inbuilt sentence splitter here and if you say something like nltk.sent_tokenize instead of word tokenize, sent tokenize and pass the string, it will give you sentences. If you count the number of sentences in this particular case, we should have four. Yey! We got four. The sentences themselves are exactly what we expect. This is the first sentence, is the first one. A gallon of milk in the US cost $2.99, that's the second one. Is this the third sentence? That's the third one. And yes it is, is the fourth one. So, what did you learn here? NLTK is a widely used toolkit for text and natural language processing. It has quite a few tools and very handy tools to tokenize and split a sentence and then go from there, lemmatize and stem and so on. It gives access to many text corpora as well. And these tasks of sentence splitting and tokenization and lemmatization are quite important preprocessing tasks and they are non-trivial. So you cannot really write a regular expression in a trivial fashion and expect it to work well. And NLTK gives you access to the best algorithms or at least the most suitable algorithms for these tasks. </td>
    <td></td>
  </tr>
</table>


3. [Advanced NLP tasks with NLTK](https://www.coursera.org/lecture/python-text-mining/advanced-nlp-tasks-with-nltk-wWEVW)

<table>
  <tr>
    <td>
In this video, we are going to move on from basic NLP tasks to advanced NLP tasks using NLTK. 

If you recall the NLP tasks that we look so far are counting words, counting frequency of words, finding unique words, finding sentence boundaries, even finding tokens in stemming. 

In this video, we will talk about part of speech tagging and parsing the sentence structure. But other NLP tasks like semantic role labeling and named entity recognition, that we'll cover later on. 

Let's start with part-of-speech tagging, or POS tagging. 

Recall from your high school grammar that part-of-speech are these verb classes like nouns, and verbs, and adjectives. 

There are many, many more tags than these. You can see that you have conjunctions and cardinals. Cardinals are, if you have a number, then you are to kind of assign that word class. You have determiner, you have prepositions, you have modal words, you have nouns. Again, nouns could be singular nouns, and plural nouns, and proper nouns. You have possessives and pronouns again of multiple types, you have adverbs, symbols, and then verbs. And verbs themselves are multiple classes, you have verbs and gerunds and past tense verbs and so on. 

How to get them? Recall that in NLTK, you need to import NLTK first, and then you can get more information about these word classes, by issuing a help command. So nltk.help you've been tagset, this comes from upenn tags. So you, if you say upenn_tagset, and then give the tag there, MD, it'll tell you that MD stands for modal auxiliary, and these are the words that are modal words. Can, cannot, could, couldn't, may, might, ought, shall, should, shouldn't, will, would, and so on. 

So how do you do POS tagging with NLTK? Recall that you have to split a sentence into words, okay. 

So this example is something that we have seen. Children shouldn't drink a sugary drink before bed. But if I split it into words, so you could use a word_tokenize from an NLTK package that text13 then becomes this individual word. Recall that there were ten words here, children shouldn't, as in a separate token, and then drink a sugary drink before bed, and the full stop is the last token. 

Then, if you run the post tag there, by using this command nltk.pos_tag on this tokenized form, you'll get the tags, so children is a plural noun, should is a model word, n't is tagged as an end verb, drink is a verb, a is a determiner, sugary is an adjective and so on. So that is how you'll get the part of speech. Now, why do you want part of speech, because if you are trying to club all nouns together because they all have some sort of a similar form or all modulus together then using the part of speech tag gives you one class or one cluster for all of these words, and then you don't have to address them individually. So when your doing some feature engineering or feature extraction that is very useful. We'll talk about features and how to use it next week. 

Now, that we know how to do POS tagging, we should talk about ambiguity in POS tagging. 

In fact, ambiguity is very common in English. Let's take this example, visiting aunts can be a nuisance. What do you think it means? 

Does it mean visiting aunts can be a nuisance, and this going to visit aunt can be nuisance or aunts who are coming in are nuisance. 

So if you tokenize the word with visiting aunts can be a nuisance and do a pos_tag on that you'll get one tag. So when you say visiting is a verb, and that aunts is a noun, a plural noun. 

This representation shows that you are doing the act of visiting. So visiting is the verb for the aunt, you are going to visit your aunt. 

The alternate POS tag would be visiting as an adjective for aunt. 

So that would mean that the aunts are the ones who are visiting you. 

So you can see that this particular sentence is ambiguous, the way it is written. 

POS tagging is not, and if fact, NLTK gives the first version and not the second. And that is because you don't really have a way to say, give me all possible variance. And sometimes the ones that take precedence are the ones where you mostly likely see the word visiting in the context of usages in a large corpus. So visiting is more often used as a continuous or a form of a verb rather than using it as an adjective. 

So the probability of finding visiting as an adjective is lower and so the first word winds up, but this is a valid alternate representation of POS tag. 
6:08
Okay, so that was POS tagging. 

Once, we have done some POS tagging, we can look at parsing the sentence structure itself. Okay, now, that we have looked at how you would find out the parts of speech of a sentence. Let's look at the sentence structure itself, and parsing of the sentence structure. 

Making sense of sentences is easy if you follow a well defined grammatic structure, okay? So when you have a sentence like this, Alice loves Bob, 

you want to know which is the noun and which is the verb and so on. But we also want to know how are they related in the sentence? How did they come together to form the sentence? So in NLTK, you'll use nltk.word_tokenize Alice loves Bob. That'll give you these three words in the sentence, right, Alice loves Bob. Those are the three words, but the sentence itself is constituted of two things. You have noun phrase and verb phrase. The word phrase itself can be a word followed by a noun phrase. 

That is the grammar extraction in English. 

And for this particular sentence this is the grammar structure that is used. Now, that we have that, noun trace itself can be a noun, so Alice can be a noun and then Bob is the other noun and there is one verb that is loves. 

So by doing this structure and writing out this grammar where you have these grammar rules on the right saying S gives NP VP, VP is V and NP, and then Alice and Bob can be NP and loves would be a V. You have created what is called a context free grammar. 

And then you can use NLTK's contextual grammar input statements like nltk.CFG.fromstring, and then you write the string out, that will give you the grammar. 

And you can use this grammar to parse the sentence. So you can use nltk.ChartParser, you create a parser using the grammar that you have defined, sort is as parser, and then parse the sentence. So you parse text15, and when you parse it, it gives you parse trees. And then you can print that tree, so you'll print the tree as this. So the way you will parse it out, for lack of a better word, is you have S, that's sentence, that constitutes two brackets. The NP bracket that has Alice and a VP bracket where a VP bracket has a V bracket for loves the verb and an empty bracket for Bob exactly the way we have drawn the tree. 

As with part of speech tagging parsing also is ambiguous and ambiguity may exist even if sentences are grammatically correct. So let's take this example, this is a very famous example. I saw the man with a telescope. 

Now, did you see with the telescope? Or did you see the man who was holding a telescope? 

There are two meanings, and the meaning is with respect to where this preposition with the telescope gets attached. So this is a typical preposition attachment problem. 

And if you look at the grammar, you will see where that ambiguity comes in. The fact that this is grammatically correct is because you have sentence that is noun phrase and verb phrase. But then this verb phrase can be split into two things. So noun phrase is I, that is clear. But verb phrase could either be a verb followed by a noun phrase as in, saw the man with the telescope. 

Or it could be a verb phrase followed by a preposition phrase, where you have the verb phrase is, saw the man, because verb phrase could still be verb and a noun phrase, that's what the saw the man is. And then you have preposition phrase, which is preposition in a noun phrase, so with and the telescope. 

So these are the two alternatives, the bold red and the dotted red. And those are the two trees that distinguish the two meanings of you would parse out the meaning from the sentence. 

We can do the same thing and this becomes apparent when we use NLTK's parsing as well. So you first tokenize the sentence, nltk.word_tokenize I saw the man with a telescope. 

And you can load a grammar, so if write your own file mygrammar1.cfg that has these lines. That the sentence is non-present verb phrase. Then verb phrase could be a verb and a noun phrase or a verb phrase and a preposition phrase. Preposition phrase itself can be a preposition and a noun phrase and so on. 

And then if you load this up as a grammar using nltk.data.load, that will create a grammar. Then you can create, the grammar has 13 rules, 13 productions, that is what it is called. And then you can create a chart parser using this grammar, so you can say nltk.ChartParser{grammar1), exactly the same way we did it a few slides back. 

And then print out all trees from this parser from that I sorry the parser you will see that there are indeed two trees. One, which is a noun phrase and a verb phrase where verb phrase has verb phrase and a preposition phrase. And another one which is a noun phrase and a verb phrase where verb phrase is a verb and noun phrase. And the noun phrase itself has a preposition with it. 

So these are the two parse tree structures that you'll get when you parse using this context with grammar. 

Now, we gave examples of simple grammars, and said we'll create a context for grammar out of it, but we can not do that every time. In fact, generating the grammar and generating grammar rules itself is a learning task that you could learn, and you need a lot of training data for that. And a lot of manual effort and hours have gone into creating what is known as a tree back, basically, a big collection of parse trees. From Wall Street Journal, and in fact, if we have access to treebank through NLTK. So if you say from NLTK corpus import treebank and say treebank.parsed_sentences this particular first sentence from Wall Street Journal. And you print out, you will see the sentence and you'll realize that you have seen it before. This is that Pierre Vinken sentence. So, Pierre Vinken, 61 years old, will join the board as a nonexecutive director, November 29th. So that sentence has been parsed using this structure, in the tree bank, and you have that available. 

Just to conclude, you see that there are complexities in part of speech tagging and parsing, beyond just how to do it. So for example, there is this usage and uncommon usage of words. An example is the old man the boat, when you read it you feel that the old man is actually a man who is old, but then you cannot finish the sentence, and parse it properly. The current parse would make man the verb, that is to man something. 

However, when you do a word tokenize on the sentence do a post stack, 

you'll get man as a noun, and old as an adjective. And this particular sentence is not grammatically correct, there's no parse string for this structure. 

Sometimes even well formed sentences that have parse structures may still be meaningless, so there is no semantics or meaning associated with that. The great example is, colorless green ideas sleep furiously. When you read this you can see that the sentence structure seems right, 

but it does not make any sense, meaningless. 

And that is because when you do, when you find out that the word tokens will be word tokenize and the part of speech tag on that, you will see that it doesn't really do a good job, it says, colorless is a proper noun, and then green is an adjective, rather than saying colorless and green are both adjectives, there is an error there. But even if you remove the word colorless, say, green ideas sleep furiously, you would have the perfect post tag there. Green is an adjective, ideas is a noun, sleep is a verb, and the you have furiously, that is an adverb, and this particular order is perfectly fine. But it still doesn't make any sense meaning wise. So there are many more layers of complexity in language that parse trees and apart of three stacks don't address so far, okay? So to conclude all of the take home concepts here, we looked at POS tagging and saw how it provides insight into the word classes, and word types. In the sentence, parsing the grammatical structures helps derive meaning. 

Both tasks are difficult, and there is linguistic ambiguity that increases the difficulty even more. And you need better models, and you can learn those using supervised learning, 

NLTK provides access to these tools and also has data, in terms of tree bank for example, that could be used for training. 

Next module, we're going to go into more detail about how do you train these models, how do you build a supervised method, and supervised technique, for these. But that's for another time. 
Explore our Catalog
Join for free and get personalized recommendations, updates and offers.

</td>
    <td></td>
  </tr>
</table>


# **MODULE 3**

## #DONE Text Classification

<table>
  <tr>
    <td>Transcript</td>
    <td>Tóm tắt</td>
  </tr>
  <tr>
    <td>
Welcome back. In this module, we are going to
talk about text classification or supervised learning for text. To start and set up the case,
I want you to look at this paragraph. It is about a medical document. And I want you to think about
the specialty that this relates to. Is it nephrology, neurology, or podiatry? Looking at the words,
you would think that it's podiatry, because you have foot there, you have something to do with fungal
infection and skin infection, and so on. And, nephrology,
which is the science of kidneys or neurology that is about the brain,
This is more closely related to the study of foot, so
that's podiatry, right? Now, given the three classes that
you have already, that's nephrology, neurology and podiatry,
let's look at another paragraph. Here, you will see that it
talks about kidney failure. And just by looking at the first few
words you would know that this probably belongs to nephrology. Now think about how you
made that decision. Did you use the work nephrology anywhere? It's not there in text. Then how did you know
that it's nephrology? The icon is important, but when you
are talking about text classification, you don't have that icon. I just put it up there. So you're looking at the words, so kidney
failure or renal failure and you somehow magically know that nephrology relates
to kidneys and renal diseases and so on. This is an important characteristic
of identifying a class based on text. And what he just did is classification. So what is classification? You have a given set of classes, in this particular case we
had these three classes. And note that I don't even have to tell
you the name nephrology or neurology. I could just give them class one,
two, three or these three icons. But you know that they
are these three concepts, these three classes that you
want to classify them into. And then the task is to assign
the correct class label to a given input. There are a lot of examples of text
classification when you look at it. For example,
if you are looking at a news article, and depending on which page of
the newspaper it belongs to, you would want to categorize it as
politics, sports, or technology. There are many more classes, of course. But in this case, let's say we want to distinguish them
into one of these three classes. Or you get an email and you want to label
that email as a spam or not a spam. Should it go into spam folder? How does any mail client decide
that it should go in this category? Basically, what is happening is there is
a text classification model running time. Think of sentiment analysis. You read a movie review and
just by reading it, you want to decide if it's a positive
review or a negative review. The text classification can
actually be at very scales. All of these are really at the scale
of a document, and you could call a paragraph a document, or a news report
a document, or an email a document. But you could also have text
classification at a word level. So think of the problem
of spelling correction. And you have weather written two ways. One means the climate,
the other is a construct for the sentence. Which is the right spelling when
you are writing a sentence? The weather is great. Is it the first one or the second one? What about the word color and the way that it's spelled in
British English and American English? Depending on all the other words, and if
suppose you're writing a BBC news article, or reading one, you are most
likely to see the second spelling. The first spelling in that
context would appear as an error. You need to understand a correct for
those. That would be the spelling
correction problem. All of these tasks are in general what
are known as supervised learning tasks. It's supervised because just like
humans learn from past experiences, machines learn from past instances. So, for example,
in a supervised classification task, you have this training phase
where information is gathered and a model is built and an inference
phase where that model is applied. So, for example, you'll have a set of
inputs that we call the labeled input, where we know that this
particular instance is positive, this one is negative, and so on. So in this case, let's just do examples. Green and red, or light and
dark, or positive and negative. And you did that set that
the label set up instances and feed it into a classification algorithm. This classification algorithm
will learn which instances appear to be more positive than negative
and build a model for what it learns. Once you have the model,
you can used it in the inference phase, where you have unlabeled input and
then this model will take it and give out labels for those inputs,
for those input instances. In this supervised learning, you learn
a classification model on properties. So when we say instances,
you're really looking at properties, or features of these instances, and
the model that is learned is basically importance of sort, or weight,
that is given to those properties. And all of this is basically learned
from the labeled instances given to you. More formally, the set of attributes or
features that represents the input is denoted by x, usually written
as bold x, because it's a vector. It is a set of individual features,
and in this particular case let's say there are n features.What
would these n features be? Let's take an example of email and
in order to take with it spam or not, you would say where does it come from? Does it have Kind of interesting
words like Nigeria or prince or deposit money or
something like that? So those would be attributes on which
you make a decision whether this email should go in the spam folder or not. Then, you have the class label,
the set of class labels is Y. Let's say there are k classes. If it's just two classes like positive and
negative, then k will be 2. If it was this medical speciality
example we saw where it's nephrology, neurology or podiatry, then k is 3. One of those classes is
the class label assigned to the instance, and that is by small y. Once we have learned this
classification model, we apply that model to new
instances to predict the label. So when we look at these, there are some terminology of data
sets that you would see very commonly. So again, there are two phases, the
training phase and the inference phase. The training phase has labeled data set. And in general, in the inference
phase you have unlabeled data set. Unlabeled data set is where
you have all instances and you have the x defined,
but you don't have a y. You don't have a label. Whereas in the labeled data set,
you have the x and the y for every instance given to you. However, in training, you don't use the
entire label set for training purposes. Because then you will not
know how well your model is. So what you want to do is to use
a part of it as training data where you actually learn parameters. Learn the model, but
leave some aside as a validation data set, or it's sometimes called
hold out data set. So that in the training phase,
you can learn on the training data but then test, or evaluate, or
set parameters on the validation data. And then, you want another data set
to really test how well you do. We are to never use it in training. You don't set your
parameters based on that. But you just evaluate on that,
so that you can judge whether the model was really good or
not on completely unseen data. You have seen all of these concepts
in previous courses within this specialization that goes straight. But I want to kind of bring them here so that we have the context in
which we're going to talk about. One last thing about classification and
that's classification paradigms. We talked about cases when the set
of possible labels are two, right? Positive and negative or green and red or
yes and no or spam or not spam, right? All of these tasks are called
binary classification tasks because the number of possible classes is two. When that increases, When the number
of examples is more than two, number of classes I'm sorry,
is more than two, it's called multi-class
classification problem. And in some instances, you might want
to label with more than one labels. And when that happens, when the data and
classes are labeled by two or more labels,
that is called multi-label classification. Typically, we will look at
binary classification or multi-class classification, but there
are some instances within this module and the scores where we will look at
multi-label classification too. In general,
when we talk about classification, we're talking about binary or
multiclass classification. When you look at what questions to ask
in a supervised learning scenario, in the training phase, the questions that you
need to answer are what are the features? How do you represent them? How do you represent
the input given to you? What is a classification model or
the algorithm you're going to use? And what are the model parameters,
depending on the model you use? These are the questions that you will
answer while you're building your model. You need to know how do
you represent input? How are they going to train, or
what model are they going to train, and then what is the output
of that training process? And in the inference phase, you need to define what
are the expected performance measures? What is a good measure? How do you know that you
have built a great model? What if a performance measure
you'll use to determine that? And these are the questions you would
answer when you are building a supervised learning model. The same thing applies to text. And in the next few videos we
are going to answer these one by one.</td>
    <td>Giảng viên đưa ra 2 văn bản và hỏi xem là văn bản đấy nói về thận học, thần kinh học hay nói về bệnh chân.
Sau đó thầy giáo đưa ra định nghĩa về bài toán phân lớp:



Một số bài toán phân lớp thực tế:
Topic Identification: Ví dụ như báo mới
Spam detection: Nhận dạng email, bài đăng là spam hay không ( Phân lớp nhị phân) 
Sentiment Analysis: Ví dụ như nhận dạng một bình luận, 1 tin nhắn là tích cực hay tiêu cực ( Phân lớp nhị phân )
Spelling correction: Weather hay whether, color hay colour







Supervised Learning:
Là học từ những dữ liệu được gán nhãn trong lịch sử, giống như cách con người học từ những trải nghiệm trong quá khứ.



Bài toán supervised Classification:




Bài toán phân lớp giám sát có thể được chia làm 3 loại:
Phân lớp nhị phân: Y = 2. Ví dụ như lớp tiêu cực, tích cực
Phân lớp đa lớp (multi-class classification): Y > 2. Ví dụ phân lớp sản phẩm vào danh mục, số danh mục > 2
Phân lớp đa nhãn (multi-label classification): Ví dụ như một bài báo vừa có thể thuộc chuyên mục giải trí, vừa có thể thuộc chuyên mục đời sống.

Để giải quyết bài toán Supervised Learning thì ta cần trả lời những câu hỏi sau:

Ở pha huấn luyện:

Các feature ở đây là gì, làm sao để biểu diễn chúng. Ví dụ như với text thì có thể là word2vec, bag of words
Lựa chọn mô hình nào để phân lớp: ví dụ như cây quyết định, Naive Bayes, ...

Ở pha dự đoán:

Độ chính xác kỳ vòng là gì, sử dụng độ đo nào để đo độ chính xác (f1, accuracy, …) 

</td>
  </tr>
</table>


## #DONE Identifying Features from Text 

<table>
  <tr>
    <td>Transcript</td>
    <td>Tóm tắt</td>
  </tr>
  <tr>
    <td>In this video we are going to answer
the first of the three questions on what to think about in
a supervised learning scenario. And that is feature identification. How do you identify features from text? So why is text data so unique? The text data presents very
unique set of challenges when you're looking at in
a supervised learning scenario. All the information that you need or that
you have in these cases is all in text. But text is a very weird concept. There are different ways you
can parse a text document. The features can be pulled from
text in different granularities. So let's take an example of the type
of textual features that you will get. The basic constructs in
text is the set of words. So, these are by far,
the most common class of features and add a significant number of
features when you add to it. So, for example, in English language,
there are about 40,000 unique words. So just looking at that in common English,
you would have 40,000 features. If you're looking at social media or
some other genres of data, you might get very many more
number of features because you could have unique word spellings and
so on and all of those would be words. When you get these many features, one of
the questions you're to start answering is, how do you handle
commonly-occurring words? In some cases, they are called stop words. Words like the,
that occurs fairly commonly, and that's the most frequent
word in the English language. And every document, every sentence
would probably have the word the. But more generally the word
the is not as important for a classification task than any other word. Let's say if you're talking about politics
and you have parliament as a word. Parliament as a word is more
important than the word the to determine whether the document
belongs to the politics class. The next step is normalization. Do you make all the words lower case so
that parliament with a capital P and a small p are treated the same,
or add the same feature? Or should we leave it as-is? US, capitals, would be the United States. Whereas if you make it lowercase, it would
be indistinguishable from the word us. So, in some cases, you want to leave it as
is, in some cases you want it lowercase, so how do you make that choice? There are also issues about stemming and
lemmatization. So for example, you don't want
plurals to be different features. So you would want to lemmatize them or
to stand them. Then, all of this is still about words. Let's go beyond words. You can actually identify features or
characteristics of words. For example, capitalization. The word, as I said, US is capitalized, White House, with the W capitalized and
the H capitalized, it's very different than white house,
a white house, right? So, capitalization is important feature to
identify certain words and their meanings. You could use parts of speech
of words in a sentence. And those could be a feature. So for
example if it's important to say that this particular word had
a determinant in front. Then that word is, so
then that becomes an important feature. An example would be the weather,
whether example. Recall that we talked about a spelling
correction problem where you want to determine whether the correct spelling
is whether as in W-H-E-T-H-E-R or weather as in W-E-A-T-H-E-R. If you see a determiner like,
the, in front of that word, it most likely would be
the weather as in W-E-A-T-H-E-R. So that particular part of
speech before this word becomes a very important feature. You also might want to know
the grammatical structure or parse the sentence and
get the sentence parse structure to see what is the verb associated
with a particular noun. How far is it from the associated noun and
so on. And then, you may want to group
words of similar meaning to have one feature to represent a set of words. An example would be buy,
purchase, and so on. These are all synonyms, and you don't
want to have two different features, one for buy, one for purchase, you may
want to group them together because they mean the same,
they have the same semantics. But it could also be other groups,
like titles or honorifics, like Mr, Ms,
Dr, Professor, and so on. Or the set of numbers, or digits, because
you don't want to specifically have a feature for zero and other features for
one, and so on, all the numbers. So you might want to say, if it's
a number anywhere between 0 to 10,000, I'm just going to call it something,
a number, right, so suddenly you've reduced
10,000 features into one. Or the same with dates. If you are able to recognize dates
using irregular expressions for example and you do a very good job of it,
then you may want say, you know what, maybe all dates would be identified and
called one feature as a date, because I don't want to learn something that is for
every individual date possible. Then it's an infinite list. Other type of features would be
depending on the classification tasks. So for example, you may have features
that come from inside the words or have features with word sequences. An example would be bigrams or trigrams. The White House example comes to mind. Where you want to say, White House, as a two word construct as a bigram
is conceptually one thing. As compared to white and house as two
different features, two different things. You may also want to have character
sub-sequences such as ing or ion. And just by looking at
it you know that ing is basically saying it is a word
in its continuous form, right? So, just looking at ing in a word,
we'd be able to call it as a verb. And ion more likely at the end of
a word would be a noun of some form. So just these character subsequences
can help you identify some classes if that is important for this particular
classification task that you have. So how would you do it? We have talked about some
of these features and I would suggest you recall
the lectures from previous week that was about natural language
processing and basic NLP tasks. And we have addressed some
of these things there. You just want to identify them now and
make it available as features for your classification task. We'll see more examples of features soon.</td>
    <td>Các lý do khiến cho dữ liệu kiểu văn bản độc đáo:
Tất cả các thông tin mà chúng ta cần đều nằm trong đoạn văn bản đó
Các đặc trưng có thể được lấy ra từ text theo các chi tiết khác nhau


Các kiểu đặc trưng của văn bản:

Words

 Ex: Ví dụ như tiếng anh có 40,000 từ, do đó 1 văn bản Tiếng Anh có thể biểu                                       diễn mỗi văn bản Tiếng Anh thành 1 vector có 40,000 chiều (40,000 đặc trưng)

Đây là đặc trưng phổ biến nhất
Cần xử lý các từ mà xuất hiện phổ biến (Stop words). Ví dụ những từ như là "The" xuất hiện trong hầu hết các văn bản Tiếng Anh, ko có giá trị trong việc phân lớp văn bản.
Normalization: Chuyển text về dạng chữ thường hoặc là không thay đổi nó.
Một số trường hợp ta sẽ chuyển hết văn bản về dạng chữ thường, một số trường hợp thì không. Ví dụ như US (United States)khi chuyển về us thì sẽ mang nghĩa khác.



Stemming / Lemmatization:

Stemming: quá trình chuyển đổi các từ của câu sang các phần không thay đổi của nó. Trong nltk có hỗ trợ nhiều thuật toán stemming khác nhau đối với tiếng anh như LancasterStemmer, PorterStemmer, …

             Ví dụ: amusing, amusement, và amused  ⇒ amus

Lemmatization: Là quá trình chuyển các từ trong câu về dạng từ điển của nó. 

             Ví dụ, cho các từ amusement, amusing, and amused, sau quá trình  Lemmatization => amuse 

Các đặc tính của word: Ví dụ như có viết hoa hay không
POS của các từ trong câu. Ví dụ như trong bài toán spelling correction, nếu từ đứng trước là “The” thì từ đứng sau khả năng sẽ là weather hơn là whether


Phụ thuộc vào từng task phân loại, đặc trưng có thể lấy ra từ các từ hay chuỗi các từ 
bigrams, trigrams, n-grams: White House
Các chuỗi con trong từ: ing, ion, … Ví dụ như các từ kế thúc bằng ing trong tiếng anh thì khả năng sẽ là động từ, kết thúc bằng ion thì khả năng cao sẽ là danh từ, ...




</td>
  </tr>
</table>


## #DONE Naive Bayes Classifiers

<table>
  <tr>
    <td>Transcript</td>
    <td>Tóm tắt</td>
  </tr>
  <tr>
    <td></td>
    <td>Bài toán classifying search queries vào các lớp: Entertainment, Computer Science, Zoology.

Giả sử có 1 query là "Python":
Nếu ta hiểu Python là con rắn => Zoology
Nếu ta hiểu Python là ngôn ngữ lập trình => Computer Science
Nếu ta hiểu Python như là Python Monty ( Tên nhóm hài kịch) => Entertainment

Giả sử query là “Python Download” => Lớp phù hợp nhất sẽ là Computer Science



Mô hình xác suất:

p(y= Entertainment), p(y=CS), p(y=Zoology) là xác suất tiền nhiệm ( Prior Probability)

p(y=Entertainment|x=Python) là xác suất hậu nghiệm (Posterior Probability).

Công thức Bayes là:
Posterior = Prior * Likelihood / Evidence




Giả định “Ngây thơ” (Naive assumption): Khi cho nhãn y, các feature sẽ độc lập lẫn nhau





Cách tính Pr(y) theo MLE


Giải quyết trường hợp P(xi|y) = 0 bằng smoothing. Có rất nhiều cách làm smooth trong đó có một cách phổ biến là dùng Laplace smoothing


Về MLE và các kiểu Gaussian Naive Bayes, Bernoulli Naive họ không nói rõ nhưng có thể đọc trong link này để hiểu thêm: https://mattshomepage.com/articles/2016/Jun/07/bernoulli_nb/</td>
  </tr>
</table>


# MODULE 4

## # DONE Text similarity

Ứng dụng của semantic similarity là :

* Group các từ tương đồng vào một khái niệm ngữ nghĩa

* Các task thường ứng dụng semantic similarity:

* Textual entailment

* Paraphrasing

Một trong những công cụ để giải quyết bài toán này là Wordnet, wordnet có thể hiểu là một mạng từ được tổ chức phân cấp, chức các thông tin liên quan đến ngữ nghĩa như từ đồng nghĩa

Mỗi một POS sẽ có 1 dummy root, ví dụ N sẽ có 1 dummy root, Verb sẽ có một dummy root.

Các semantic similarity measures:

* Sử dụng đường đi ngắn nhất:

Path similarity: 1 / (distance + 1) trong đó distance là đường đi ngắn nhất giữa 2 node

![image alt text](image_0.png)

* Sử dụng Lowest common subsumer (LCS):

LSC là tổ tiên gần nhất với 2 khái niệm

Độ tương đồng Lin:

![image alt text](image_1.png)

Để tính được P thì ta cần một tập dữ liệu văn bản rất lớn

## Topic Modeling

