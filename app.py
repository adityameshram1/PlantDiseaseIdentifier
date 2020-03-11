import numpy as np
from flask import Flask, request, render_template
from werkzeug import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras.backend import clear_session
import tensorflow

app = Flask(__name__)
index_page=""

@app.route('/')
def home():
    return render_template('index.html')
   
    
def prepare(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

@app.route('/result/')
def about():
    #return render_template('result'+str(index_page[0])+'.html')
    return render_template(index_page+'.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    global index_page
    if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename('image.jpg'))
      print('file uploaded successfully')
      img = image.load_img('image.jpg', target_size=(256, 256))
      x = image.img_to_array(img)
      x = x/255
      ret_val=np.expand_dims(x, axis=0)
      #result=""
      #result = model.predict_classes([ret_val])
      with graph.as_default():
          feature = model.predict([ret_val])
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    print(type(feature))
    print(feature)
    #result=np.where(feature == np.amax(feature))
    print("================================")
    #print(result)
    #print(max(result))
    index=np.where(feature[0]==np.amax(feature[0]))
    #max_value=np.max(feature)
    #return render_template('index.html', prediction_text='Disease detected: {}'.format(Classes[int(result[1][0])]))
    print(index)
    #index_page=index[0]
    index_page=Classes[int(index[0])]
    return render_template('index.html', prediction_text='Disease detected: {}'.format(Classes[int(index[0])]))    


if __name__ == "__main__":
    clear_session()
    model=load_model('crop.h5')
    Classes=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    global graph
    graph = tensorflow.get_default_graph()
    app.run(debug=True)