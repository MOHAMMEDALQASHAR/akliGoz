with open('web_face_recognition/static/js/dashboard.js', 'r', encoding='utf-8') as f:
    js_content = f.read()

print('=== Function Verification ===')
print('saveFace exists:', 'async function saveFace' in js_content)
print('deleteFace exists:', 'async function deleteFace' in js_content)
print('editFace exists:', 'async function editFace' in js_content)

print('\n=== Event Listeners ===')
print('addFaceForm listener:', "addFaceForm.addEventListener('submit', saveFace)" in js_content)
print('delete buttons listener:', ".addEventListener('click', deleteFace)" in js_content)
print('edit buttons listener:', ".addEventListener('click', editFace)" in js_content)

print('\n=== File Info ===')
print('Total characters:', len(js_content))
print('Total lines:', js_content.count('\n'))

# Check for editFace function details
if 'async function editFace' in js_content:
    print('\neditFace function FOUND')
    # Find the function
    start = js_content.find('async function editFace')
    if start != -1:
        # Get next 300 chars
        snippet = js_content[start:start+300]
        print('First part of editFace:')
        print(snippet[:200])
else:
    print('\nWARNING: editFace function NOT FOUND')
