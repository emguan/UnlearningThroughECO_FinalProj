class SlidingWindowPromptClassifier:
    def __init__(self, model_name, batch_size=8, max_length=512, stride=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride

    def classify(self, prompt):
        encoding = self.tokenizer(
            prompt,
            return_overflowing_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            stride=self.stride,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.model.device)
        attention_mask = encoding["attention_mask"].to(self.model.device)

        self.model.eval()
        scores, labels = [], []

        with torch.no_grad():
            for i in range(0, input_ids.size(0), self.batch_size):
                outputs = self.model(
                    input_ids=input_ids[i:i+self.batch_size],
                    attention_mask=attention_mask[i:i+self.batch_size]
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                batch_scores, batch_preds = torch.max(probs, dim=-1)
                for pred, score in zip(batch_preds, batch_scores):
                    label = 1 if f"LABEL_{pred.item()}" == "LABEL_1" else 0
                    labels.append(label)
                    scores.append(score.item())

        best_idx = torch.argmax(torch.tensor(scores))
        return labels[best_idx], scores[best_idx]
