document.getElementById('summarization-form').addEventListener("submit", async (e) => {
    e.preventDefault();

    // Get references to elements
    const dialogueInput = document.getElementById('dialogue-input')
    const summaryText = document.getElementById('summary-text')
    const submitButton = document.getElementById('button')
    
    const dialogue = dialogueInput.ariaValueMax.trim();
    if (!dialogue) return;

    // show processing message and disable button
    summaryText.innerText = "Processing...";
    submitButton.disabled = true;

    try {
        // send dialogue to backend
        const response = await fetch('/summarize/', {
            method: "post",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dialogue }),
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const data = await response.json();
        summaryText.innerText = data.summary || "No summary returned."
    }
    catch (err) {
        summaryText.innerText = `Error: ${err.message}`;
    }
    finally {
        submitButton.disabled = false;
    }
})