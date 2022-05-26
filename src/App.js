import logo from './logo.svg';
// import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';

import padSequences from './helper/paddedSeq'

import './App.css';
import { useEffect, useState } from 'react';




function App() {

  const url = {

    model: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
};

const OOV_INDEX = 2;

const [metadata, setMetadata] = useState();
const [model, setModel] = useState();
const [testText, setText] = useState("");
const [testScore, setScore] = useState("");
const [trimedText, setTrim] = useState("")
const [seqText, setSeq] = useState("")
const [padText, setPad] = useState("")
const [inputText, setInput] = useState("")


async function loadModel(url) {
  try {
    const model = await tf.loadLayersModel(url.model);
    setModel(model);
  } catch (err) {
    console.log(err);
  }
}

async function loadMetadata(url) {
  try {
    const metadataJson = await fetch(url.metadata);
    const metadata = await metadataJson.json();
    setMetadata(metadata);
  } catch (err) {
    console.log(err);
  }
}

const getSentimentScore =(text) => {
  console.log(text)
  const inputText = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
  setTrim(inputText)
  console.log(inputText)
  const sequence = inputText.map(word => {
    let wordIndex = metadata.word_index[word] + metadata.index_from;
    if (wordIndex > metadata.vocabulary_size) {
      wordIndex = OOV_INDEX;
    }
    return wordIndex;
  });
  setSeq(sequence)
  console.log(sequence)
  // Perform truncation and padding.
  const paddedSequence = padSequences([sequence], metadata.max_len);
  console.log(metadata.max_len)
  setPad(paddedSequence)

  const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);
  console.log(input)
  setInput(input)
  const predictOut = model.predict(input);
  const score = predictOut.dataSync()[0];
  predictOut.dispose();
  setScore(score)  
  return score;
}

useEffect(()=>{
  tf.ready().then(
    ()=>{
      loadModel(url)
      loadMetadata(url)
    }
  );

},[])

  return (
    <div>
      <textarea 
        onChange={(e)=> setText(e.target.value)}
        defaultValue=""
        value={testText}
      ></textarea>

    {testText !== "" ?
              <button style={{width:"20vh", height:"5vh"}} variant= "outlined" onClick={()=>getSentimentScore(testText)}>Calculate</button>
      : <></>}

      {testScore !== "" ?
      <p>{testScore}</p>
      :<></>}

    </div>
  );
}

export default App;
