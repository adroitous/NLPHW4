from collections import Counter, namedtuple
import math
import pprint
import nltk
import numpy
import sklearn
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
from textblob.classifiers import NaiveBayesClassifier


def buildxmltree(file):
	root = ET.parse(file).getroot()
	#every lists in this list is an instance that has been lemmatize
	textlisting =[]
	answerlist = []
	for corpus in root:
		for instance in corpus:
			text = []	
			context = instance.find('context')
			#first word after context
			text.append(context.text.strip().lower())
			for word in context:
				#tags/part of speech
				#the word following
				#fix stemming issue
				#print word.tail
				text.append(word.tail.strip().lower())
			textlisting.append(text)	
			answer = instance.find('answer')
			answerlist.append(answer.attrib.get('senseid'))
	#print len(textlisting)
	#print len(answerlist)
	#print textlisting
 	textlisting.append(answerlist)
 	#print textlisting
	return textlisting


def buildxmltags(file):
	root = ET.parse(file).getroot()
	textlisting =[]
	tags = []
	for corpus in root:
		for instance in corpus:
			text = []	
			tag = []
			context = instance.find('context')
			text.append(context.text.strip().lower())
			for word in context:
				tag.append(word.attrib.values()[0])
				text.append(word.tail.strip().lower())
			textlisting.append(text)	
			tags.append(tag)
 	final =[]
 	for e in range(len(textlisting)):
 	 	final.append(zip(textlisting[e], tags[e]))
 	#print final
	return final

#word vector only
def collocational(text, n):
	textlist = []
	temp = []
	filterwords = set(nltk.corpus.stopwords.words('english'))
	filterwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','' ,'','" ','-',"' s" ])
	for list in text:
		filterlist = [i for i in list if i not in filterwords]
		Atext=[]
		for e in filterlist:
			Atext.append(lancaster_stemmer.stem(e))
		temp.append(Atext)
	text = temp
	for element in text:
		individualtextlist = []
		for i in range(2*n+1):
			#print element.index('bank')
			i = i-n
			individualtextlist.append(element[element.index('bank')+i])
		textlist.append(individualtextlist)
	#print textlist
	return textlist

def tags(text, n):
	textlist = []
	temp = []
	filterwords = set(nltk.corpus.stopwords.words('english'))
	filterwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','' ,'','" ','-',"' s" ])
	for list in text:
		filterlist = [i for i in list if i[0] not in filterwords]
		Atext=[]
		for e in filterlist:
			Atext.append((lancaster_stemmer.stem(e[0]),e[1]))
		temp.append(Atext)
	text = temp
	for element in text:
		individualtextlist = []
		for i in range(2*n+1):
			i = i-n
			individualtextlist.append(element[[y[0] for y in element].index('bank')+i])
		textlist.append(zip(*individualtextlist)[1])
	#print textlist
	return textlist



#create vocabulary
def createvocabulary(text, n):
	vocabulary = {}
	for textlist in text:
		for element in textlist:
			if vocabulary.has_key(element):
				vocabulary.update({element:vocabulary.get(element)+1})
			else:
				vocabulary.update({element:1})
	vocabulary.pop('bank')
	tops = sorted(vocabulary.items(), key=lambda x:x[1], reverse=True)
	print tops[:n]
	return tops[:n]

def buildVectorOccurance(alist, vocab):
	dic = {}
	for e in range(len(alist)):
		dic.update({vocab[e]:alist[e]})
	return dic

def co_occurance(text, vocab):
	countlist = []
	for element in vocab:
		#print element
		if element in text:
			countlist.append(1)
		else:
			countlist.append(0)
	return 	buildVectorOccurance(countlist, vocab)

def sense_dictionary(textlist, answer):
	vocab_set = []
	index =[]
	for element in set(answer):
		indices = [i for i, x in enumerate(answer) if x == element]
 		vocab = [] 
 		for x in indices:
			vocab.append(textlist[x])
		vocab_set.append(vocab)	
	#print "\n"
	#print vocab_set
	return vocab_set

def buildVectorTag(alist):
	dic = {}
	pos = "tag_"
	#print len(alist)
	for e in range(len(alist)):
		dic.update({pos+str(e):alist[e]})
	#print dic
	return dic

def buildVector(alist):
	dic = {}
	pos = "pos_"
	for e in range(len(alist)):
		dic.update({pos+str(e):alist[e]})
	return dic

def calculateF1(correctanswer, result):
	set_correctanswer = set(correctanswer)
	averagelist = []
	F1 = 0.0
	for e in set_correctanswer:
		finalvalue = []
		resultvalue = []
		for element in result:
			if element == e:
				finalvalue.append(1)
			else:
				finalvalue.append(0)
		for element in correctanswer:
			if element == e:
				resultvalue.append(1)
			else:
				resultvalue.append(0)
		F1 += sklearn.metrics.f1_score(finalvalue, resultvalue)
	print F1/float(len(set_correctanswer))


if __name__ ==  "__main__":
	tree = buildxmltree("bank.ntrain.xml") 
	#print tree
	text = tree[:(len(tree)-1)]
	answer = tree[-1]
	testtree = buildxmltree("bank.ntest.xml")
	testtext = testtree[:len(testtree)-1]
	testanswer = testtree[-1]
	testword = collocational(testtext, 2)
	test_set = zip(testtext,testanswer)
	word = collocational(text, 2)

	textlist = []
	for x in testword:
		textlist.append(buildVector(x))
	test_set = zip(textlist, testanswer)

	diclist = []
	for x in word:
		diclist.append(buildVector(x))
	train_set = zip(diclist, answer)
	classifier = nltk.NaiveBayesClassifier.train(train_set)

	
	testtree = buildxmltree("bank.ntest.xml")
	testtext = testtree[:len(testtree)-1]
	testanswer = testtree[-1]
	testword = collocational(testtext, 2)
	test_set = zip(testtext,testanswer)

	texttag = buildxmltags("bank.ntrain.xml")
	tag = tags(texttag, 2)
	taglisting = []
	for element in tag:
		taglisting.append(buildVectorTag(element))
	tagzip = zip(taglisting, answer)
	print "+\n"
	print train_set
	print "+\n"
	print tagzip
	#classifier = nltk.NaiveBayesClassifier.train(train_set+tagzip)
	classifier = nltk.NaiveBayesClassifier.train(train_set+tagzip)
	for x in testword:
		textlist.append(buildVector(x))
	test_set = zip(textlist, testanswer)



	testtaging = buildxmltags("bank.ntest.xml")
	testtag = tags(testtaging, 2)
	testtaglisting = []
	for element in testtag:
		testtaglisting.append(buildVectorTag(element))
	#print testtaglisting
	testtag_set = zip(testtaglisting,testanswer)
	print test_set
	final = test_set+testtag_set
	print nltk.classify.accuracy(classifier, final)

	
	#print test_set
	colist = []
	for element in sense_dictionary(word, answer):
		for e in createvocabulary(element, 7):
			colist.append(e[0])
	print 

	co_occurancelist = []
	for x in word:
		co_occurancelist.append(co_occurance(x, colist))

	print co_occurancelist
	co_occurancetrain = zip(co_occurancelist, answer)
	print "co_occurancetrain"
	print co_occurancetrain
	classifier = nltk.NaiveBayesClassifier.train(co_occurancetrain)
	


	cotestlist = []
	for element in sense_dictionary(testword, testanswer):
		for e in createvocabulary(element, 7):
			cotestlist.append(e[0])
	co_occurancetestlist = []
	for x in testword:
		co_occurancetestlist.append(co_occurance(x, cotestlist))
	co_occurancetest = zip(co_occurancetestlist, testanswer)
	print nltk.classify.accuracy(classifier, co_occurancetest)	

	result = []
	correctanswer = []
	for e in co_occurancetest:
		result.append(classifier.classify(e[0]))
		correctanswer.append(e[1])
	calculateF1(correctanswer,result)
	
