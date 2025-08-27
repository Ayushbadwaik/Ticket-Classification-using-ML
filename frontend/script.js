async function classifyTicket() {
  const ticket = document.getElementById("ticketInput").value.trim();
  if (!ticket) {
    alert("Please enter a ticket text!");
    return;
  }

  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ticket})
  });

  const data = await response.json();

  if (data.error) {
    alert(data.error);
    return;
  }

  // Decide color class based on category
  let categoryClass = "";
  if (data.category === "Billing") categoryClass = "billing";
  else if (data.category === "Technical Issue") categoryClass = "technical";
  else if (data.category === "Product Inquiry") categoryClass = "product";

  // Show latest result
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = `
    <div class="result-card ${categoryClass}">
      <p>📌 Ticket: ${data.ticket}</p>
      <p>📂 Category: ${data.category}</p>
      <p>✅ Confidence: ${data.confidence}%</p>
    </div>
  `;
  resultDiv.classList.remove("hidden");

  // Add to history
  const historyDiv = document.getElementById("history");
  const historyList = document.getElementById("historyList");

  const historyItem = document.createElement("div");
  historyItem.classList.add("history-item");
  historyItem.innerHTML = `
    <strong>${data.category}</strong> (${data.confidence}%)
    <br> <span>${data.ticket}</span>
  `;

  historyList.prepend(historyItem); // latest on top
  historyDiv.classList.remove("hidden");

  // Clear textarea
  document.getElementById("ticketInput").value = "";
}

