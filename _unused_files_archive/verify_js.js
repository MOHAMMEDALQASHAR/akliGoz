// Summary Test - Check if required functions exist in dashboard.js
const fs = require('fs');

const jsContent = fs.readFileSync('web_face_recognition/static/js/dashboard.js', 'utf8');

console.log('=== Function Verification ===');
console.log('saveFace exists:', jsContent.includes('async function saveFace'));
console.log('deleteFace exists:', jsContent.includes('async function deleteFace'));
console.log('editFace exists:', jsContent.includes('async function editFace'));

console.log('\n=== Event Listeners ===');
console.log('addFaceForm listener:', jsContent.includes("addFaceForm.addEventListener('submit', saveFace)"));
console.log('delete buttons listener:', jsContent.includes(".addEventListener('click', deleteFace)"));
console.log('edit buttons listener:', jsContent.includes(".addEventListener('click', editFace)"));

console.log('\n=== File Size ===');
console.log('Total characters:', jsContent.length);
console.log('Total lines:', jsContent.split('\n').length);
