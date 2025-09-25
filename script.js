function openAbout() {
  document.getElementById("aboutModal").style.display = "block";
}



function toggleUserInfo() {
  const infoBox = document.getElementById('userInfo');
  infoBox.style.display = infoBox.style.display === 'block' ? 'none' : 'block';
}

window.addEventListener('click', function (e) {
  const userBox = document.querySelector('.user-dropdown');
  const infoBox = document.getElementById('userInfo');
  if (!userBox.contains(e.target)) {
    infoBox.style.display = 'none';
  }
});


// ✅ File Selection & Preview
function handleFileSelection(event) {
  const file = event.target.files[0];
  const allowedTypes = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/jpeg',
    'image/png',
    'image/jpg'
  ];

  const container = document.getElementById('filePreviewContainer');
  container.innerHTML = ''; // Clear previous content

  if (!file) return;

  if (!allowedTypes.includes(file.type)) {
    alert("Invalid file type. Only PDF, DOC, JPG, JPEG, PNG are allowed.");
    event.target.value = '';
    return;
  }

  const ext = file.name.split('.').pop().toLowerCase();

  // Image preview
  if (['jpg', 'jpeg', 'png'].includes(ext)) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = 'Image Preview';
    img.style.maxWidth = '250px';
    img.style.maxHeight = '200px';
    img.style.borderRadius = '8px';
    img.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.1)';
    img.style.border = '1px solid #ddd';
    img.style.marginBottom = '6px';
    container.appendChild(img);
  }

  // File name and icon
  const icon = document.createElement('i');
  if (ext === 'pdf') icon.className = 'fas fa-file-pdf';
  else if (ext === 'doc' || ext === 'docx') icon.className = 'fas fa-file-word';
  else icon.className = 'fas fa-file-image';
  icon.style.marginRight = '6px';

  const link = document.createElement('a');
  link.textContent = file.name;
  link.href = URL.createObjectURL(file);
  link.target = '_blank';
  link.style.color = '#e0e0e0';
  link.style.fontWeight = '500';
  link.style.textDecoration = 'none';
  link.style.fontSize = '15px';

  const fileLine = document.createElement('div');
  fileLine.style.display = 'flex';
  fileLine.style.alignItems = 'center';
  fileLine.appendChild(icon);
  fileLine.appendChild(link);
  container.appendChild(fileLine);
}

// ✅ Extract Text from Uploaded File
async function extractText() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  const extractedTextArea = document.getElementById('extractedText');
  const status = document.getElementById('extractStatus');


  if (!file) {
    alert('⚠️ No file selected for extraction.');
    return;
  }

  status.textContent = '⏳ Extracting text...';
  extractedTextArea.value = '';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/extract-text', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.text) {
      extractedTextArea.value = data.text;
      document.getElementById('translatedText').value = ''; // clear translation if any
      status.textContent = '✅ Extraction completed.';
      extractedTextArea.dispatchEvent(new Event('input'));
    } else {
      status.textContent = '❌ ' + (data.error || 'Extraction failed.');
    }
  } catch (err) {
    console.error(err);
    status.textContent = '❌ Extraction error.';
  }
}


// ✅ Translate Extracted Text
async function translateText() {
  const extractedText = document.getElementById('extractedText').value.trim();
  const translateStatus = document.getElementById('translateStatus');
  const targetLang = document.getElementById('languageSelect').value;

  if (!extractedText) {
    translateStatus.textContent = '⚠️ No text available to translate.';
    return;
  }

  translateStatus.textContent = '⏳ Translating...';

  try {
    const response = await fetch('/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: extractedText, target_language: targetLang })
    });

    const data = await response.json();

    if (data.translated_text) {
      document.getElementById('translatedText').value = data.translated_text;
      document.getElementById('translatedHidden').value = data.translated_text;
      translateStatus.textContent = '✅ Translation completed.';
      document.getElementById('translatedText').dispatchEvent(new Event('input'));
    } else {
      translateStatus.textContent = '❌ ' + (data.error || 'Translation failed.');
    }
  } catch (err) {
    console.error(err);
    translateStatus.textContent = '❌ Translation error.';
  }
}

// ✅ Summarize Text
// ✅ Summarize Text (Fixed)
async function handleSummarize() {
  const useTranslation = document.getElementById('useTranslation').checked;
  const extractedText = document.getElementById('extractedText').value.trim();
  const translatedText = document.getElementById('translatedText')?.value.trim() || '';
  const summaryStatus = document.getElementById('summarizeStatus');

  const textToSummarize = useTranslation ? translatedText : extractedText;

  if (!textToSummarize) {
    alert('⚠️ Please provide or extract some text before summarizing.');
    summaryStatus.textContent = '';
    return;
  }

  summaryStatus.textContent = '⏳ Summarizing...';

  try {
    const response = await fetch('/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        use_translation: useTranslation,
        text: textToSummarize
      })
    });

    const data = await response.json();

    if (data.summary) {
      document.getElementById('summaryResult').textContent = data.summary;
      summaryStatus.textContent = '✅ Summary generated.';
      document.getElementById('summaryResult').dispatchEvent(new Event('input'));
    } else {
      summaryStatus.textContent = '❌ ' + (data.error || 'Summarization failed.');
    }
  } catch (err) {
    console.error(err);
    summaryStatus.textContent = '❌ Summarization error.';
  }
}


// ✅ Download Summary as Text File
function downloadSummary() {
  const summary = document.getElementById('summaryResult').value;
  if (!summary.trim()) {
    alert('⚠️ No summary available to download.');
    return;
  }

  const blob = new Blob([summary], { type: 'text/plain;charset=utf-8' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'summary.txt';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

document.addEventListener("DOMContentLoaded", () => {
  // 🟢 Welcome Text
  const username = localStorage.getItem("username") || "User";
  const welcomeSpan = document.getElementById("welcomeUser");
  if (welcomeSpan) {
    welcomeSpan.textContent = `Welcome, ${username}`;
  }

  // 🟢 About Modal Close
  const closeAboutBtn = document.getElementById("closeAbout");
  const aboutModal = document.getElementById("aboutModal");

  if (closeAboutBtn && aboutModal) {
    closeAboutBtn.addEventListener("click", () => {
      aboutModal.style.display = "none";
    });

    window.addEventListener("click", (e) => {
      if (e.target === aboutModal) {
        aboutModal.style.display = "none";
      }
    });
  }

  // 🟢 Auto-resize for all textareas
  document.querySelectorAll('textarea').forEach(textarea => {
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    });
    textarea.dispatchEvent(new Event('input'));
  });
});
