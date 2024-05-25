// import React, { useState } from 'react';
// import FileInput from './component/inputfile';
// import ResultDisplay from './component/resultdisplay';
// import MyNavbar from './component/navbar';
// import ProfileCover from './component/cover';
// import 'bootstrap/dist/css/bootstrap.min.css';

// const App = () => {
//   const [result] = useState('');
//   const file1Label = "Raw Dataset File";
//   const file2Label = "Test file";

//   const handleFileChange = (file) => {
//     // Process the file and determine the result type
//     // Example: setResult(processFile(file));
//   };

//   return (
//     <div>
//       <MyNavbar />
//       <ProfileCover />
//       <div className="container">
//         <h1>File Upload</h1>
//         <div className="row">
//           <div className="col-md-6">
//             <FileInput label={file1Label} onFileChange={handleFileChange} />
//           </div>
//           <div className="col-md-6">
//             <FileInput label={file2Label} onFileChange={handleFileChange} />
//           </div>
//         </div>
//         <ResultDisplay result={result} />
//       </div>
//     </div>
//   );
// };

// export default App;

import React, { useState } from 'react';
import FileInput from './component/inputfile';
import ResultDisplay from './component/resultdisplay';
import MyNavbar from './component/navbar';
import ProfileCover from './component/cover';
import 'bootstrap/dist/css/bootstrap.min.css';

const App = () => {
  const [result, setResult] = useState('');
  const file1Label = "Raw Dataset File";
  const file2Label = "Test file";

  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);

  const handleFile1Change = (file) => {
    setFile1(file);
  };

  const handleFile2Change = (file) => {
    setFile2(file);
  };

  const handleUpload = async () => {
    if (file1 && file2) {
      const formData = new FormData();
      formData.append('raw_data_file', file1);
      formData.append('test_data_file', file2);
  
      try {
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          body: formData,
        });
  
        const result = await response.json();
        setResult(result);
      } catch (error) {
        console.error('Error:', error);
      }
    }
  };  

  return (
    <div>
      <MyNavbar />
      <ProfileCover />
      <div className="container">
        <h1>File Upload</h1>
        <div className="row">
          <div className="col-md-6">
            <FileInput label={file1Label} onFileChange={handleFile1Change} />
          </div>
          <div className="col-md-6">
            <FileInput label={file2Label} onFileChange={handleFile2Change} />
          </div>
        </div>
        <button onClick={handleUpload} className="btn btn-primary mt-3">Upload and Predict</button>
        <ResultDisplay result={result} />
      </div>
    </div>
  );
};

export default App;
