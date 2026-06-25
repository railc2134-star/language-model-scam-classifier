import requests
import torch
import torch.nn as nn
JINA_API_KEY="your_jina_api_key_here"
URL="server URL
TOKEN="your_token_here"
HEADERS = {"xc-token": "your_token_here"}
limit=1000
page=0
all_messages=[]
print(f"getting the data from nocodb")
while True :
    params ={"limit":limit,"offset":len(all_messages)}
    reponse=requests.get(URL, headers=HEADERS, params=params)
    if reponse.status_code !=200 :
        print(f"\n error !!! nocodb")
        break
    data=reponse.json()
    records = data.get("list", [])
    if not records :
        break
    
    all_messages.extend(records)
    print(f" {len(all_messages)} got scraper from nocodb", end="\r")
print(f"\nFinal count: {len(all_messages)} messages loaded.")
all_X=[]
all_Y=[]
JINA_URL="https://api.jina.ai/v1/embeddings"
jina_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API_KEY}"}
batch_size=100
print(f"starting jina :")
for i in range(0 ,len(all_messages), batch_size):
    batch=all_messages[i : i+batch_size]
    texts=[]
    labels=[]
    for row in batch :
        message = row.get("message","")
        texts.append(message)
        answer = row.get("label", 0)
        labels.append(int(answer) if answer else 0)
    payloed = {
        "model":"jina-embeddings-v2-base-en",
        "input": texts
    }
    jina_requast=requests.post(JINA_URL ,headers=jina_headers,json=payloed)
    if jina_requast.status_code != 200: print(jina_requast.text); break
    hallway=jina_requast.json()
    datas=hallway["data"]
    for j in range(len(datas)):
        numbers = datas[j]['embedding']
        all_X.append(numbers)
    all_Y.extend(labels)
    print(f"Progress: {len(all_X)} / {len(all_messages)}", end="\r")
scam=0
for lab in all_Y: 
    scam =scam + lab
print(f"there is {scam}words")
X_rows=torch.stack([torch.tensor(x) for x in all_X])
Y_rows = torch.tensor(all_Y ,dtype=torch.float32).view(-1,1)
indiscts=torch.randperm(X_rows.size(0))
X_rondom=X_rows[indiscts]
Y_rondom=Y_rows[indiscts]
split = int(0.8 * len(X_rondom))
X_train, X_test = X_rondom[:split], X_rondom[split:]
Y_train, Y_test = Y_rondom[:split], Y_rondom[split:]
class Brain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = nn.Linear(768, 64)
        self.dropout = nn.Dropout(0.2) 
        self.boss = nn.Linear(64, 1)
        
    def forward(self, x):
        hidden = torch.nn.functional.leaky_relu(self.agent(x), negative_slope=0.01)        
        hidden = self.dropout(hidden)
        
        y = torch.sigmoid(self.boss(hidden))
        return y
Brain_on=Brain()
creation =nn.BCELoss()
ed = torch.optim.Adam(Brain_on.parameters(), lr=0.001)
for epoch in range(500):
    ed.zero_grad()
    outpute=Brain_on(X_train)
    loss = creation(outpute, Y_train)
    loss.backward()
    ed.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
torch.save(Brain_on.state_dict(), "brain1.pth")
print("Saved brain1.pth")
Brain_on.eval()
with torch.no_grad():
    test_output = Brain_on(X_test)
    predicted = (test_output > 0.5).float()
    accuracy = (predicted == Y_test).float().mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
wrong = (predicted != Y_test)
wrong_messages_idx = wrong.squeeze().nonzero().squeeze()
print("\nWrong predictions:")
for idx in wrong_messages_idx[:10]:
    print(f"Predicted: {predicted[idx].item()} | Real: {Y_test[idx].item()}")
all_texts = [row.get("message","") for row in all_messages]
X_texts = [all_texts[i] for i in indiscts.tolist()]
test_texts = X_texts[split:]

for idx in wrong_messages_idx[:10]:
    i = idx.item()
    print(f"Message: {test_texts[i]}")
    print(f"Predicted: {predicted[i].item()} | Real: {Y_test[i].item()}")
    print("---")
