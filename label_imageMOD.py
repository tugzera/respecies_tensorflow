

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

import json
import os
import time
from decimal import *
import shutil
import decimal


def encontraArquivosEmPastaRecursivamente(pasta, extensao):
    arquivosTxt = []
    caminhoAbsoluto = os.path.abspath(pasta)
    for pastaAtual, subPastas, arquivos  in os.walk(caminhoAbsoluto):
        arquivosTxt.extend([os.path.join(pastaAtual,arquivo) for arquivo in arquivos if arquivo.endswith('.jpg')])
    return arquivosTxt



def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def apagarArquivo(arquivo):
	if(os.path.exists(arquivo)):
		nome ='C:/Users/tugzera/Documents/Projeto/program/public/dados_web/imagem/'
		nome+=arquivo.replace("C:\\Users\\tugzera\\Documents\\Projeto\\program\\public\\dados_web\\imagem_temp\\","")
		if(os.path.exists(nome)):
			print ("Arquivo já existe!")
			print ("Apagando arquivo.. \n")
			os.remove(arquivo);
		else:
			shutil.move(arquivo, 'C:/Users/tugzera/Documents/Projeto/program/public/dados_web/imagem/')
	else:
    		print("Arquivo não existe!\n")


if __name__ == "__main__":


        file_name = ""
        model_file = \
          "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
        label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = "input"
        output_layer = "InceptionV3/Predictions/Reshape_1"

        parser = argparse.ArgumentParser()
        parser.add_argument("--image", help="image to be processed")
        parser.add_argument("--graph", help="graph/model to be executed")
        parser.add_argument("--labels", help="name of file containing labels")
        parser.add_argument("--input_height", type=int, help="input height")
        parser.add_argument("--input_width", type=int, help="input width")
        parser.add_argument("--input_mean", type=int, help="input mean")
        parser.add_argument("--input_std", type=int, help="input std")
        parser.add_argument("--input_layer", help="name of input layer")
        parser.add_argument("--output_layer", help="name of output layer")
        args = parser.parse_args()

        if args.graph:
          model_file = args.graph
        if args.image:
          file_name = args.image
        if args.labels:
          label_file = args.labels
        if args.input_height:
          input_height = args.input_height
        if args.input_width:
          input_width = args.input_width
        if args.input_mean:
          input_mean = args.input_mean
        if args.input_std:
          input_std = args.input_std
        if args.input_layer:
          input_layer = args.input_layer
        if args.output_layer:
          output_layer = args.output_layer

        graph = load_graph(model_file)
        i = 1
        while i != 0:  
          x = encontraArquivosEmPastaRecursivamente('C:/Users/tugzera/Documents/Projeto/program/public/dados_web/imagem_temp/', '.jpg')
          if len(x) != 0:
              st = x[0]
              file_name = st 
              st=st.replace("C:\\Users\\tugzera\\Documents\\Projeto\\program\\public\\dados_web\\imagem_temp\\","")   	
              print ("Fazendo classificao de " + st)
              t = read_tensor_from_image_file(
                    file_name,
                    input_height=input_height,
                    input_width=input_width,
                    input_mean=input_mean,
                    input_std=input_std)

              input_name = "import/" + input_layer
              output_name = "import/" + output_layer
              input_operation = graph.get_operation_by_name(input_name)
              output_operation = graph.get_operation_by_name(output_name)

              with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                  input_operation.outputs[0]: t
                })
              results = np.squeeze(results)

              top_k = results.argsort()[-5:][::-1]
              labels = load_labels(label_file)

              p=""
              p+= "{"	
              it=4
              for i in top_k:
                  p+= '"'
                  p+= labels[i]
                  p+= '"'
                  p+= ':'
                  p+=str(results[i])
                  if it > 0:
                      p+= ','
                  print(labels[i], results[i])
                  it=it-1
              p+= "}"
              st+=".txt"
              url ="C:/Users/tugzera/Documents/Projeto/program/public/dados_web/logs/"+st;
              arquivo = open(url,'w')
              arquivo.write(p)
              arquivo.close()
              apagarArquivo(x[0])
              x = encontraArquivosEmPastaRecursivamente('C:/Users/tugzera/Documents/Projeto/program/public/dados_web/imagem_temp/', '.jpg')  
          else:
              time.sleep(1)