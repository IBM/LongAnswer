      pip install torch == 1.2.0
      pip install transformers == 2.9.0
      pip install pytorch-lightning == 0.7.6



Script to finetune BART

Normal (Answer in Paragraph)
Data in /dccstor/tuhinstor/tuhin/NQ-rank
https://github.ibm.com/Tuhin-Chakrabarty/LongAnswerGen/blob/master/examples/run_train.sh



LA having a SA (Answer as single sentences)
Data in /dccstor/tuhinstor/tuhin/NQ-rank1
https://github.ibm.com/Tuhin-Chakrabarty/LongAnswerGen/blob/master/examples/run_train.sh

(change data path)


Script to generate paragraph based answer based on likelihood from finetuned-LM
https://github.ibm.com/Tuhin-Chakrabarty/LongAnswerGen/blob/master/examples/likelihood.sh


Script to get answer based on likelihood of single sentence from finetuned-LM
https://github.ibm.com/Tuhin-Chakrabarty/LongAnswerGen/blob/master/examples/likelihood1.sh



Uncomment 68-69 if doing AMR
https://github.ibm.com/Tuhin-Chakrabarty/LongAnswerGen/blob/master/examples/lightning_base.py
