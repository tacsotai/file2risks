# file2risks

## 1. Project Overview
**file2risks** is an open-source module that analyzes browser history (and other digital traces) to detect **potential risks**.  
It is part of the larger [RiskAware](https://www.sotai.co/) project, which aims to help families use technology more safely.  

⚠️ **Early stage**: results are still rough and predictions are far from perfect. This is exactly why contributions matter!  

---

## 2. Motivation & Vision
- Parents (especially developers with kids) know the tension between **freedom and safety** in the digital world.  
- Existing parental control tools are often about **blocking and surveillance**.  
- We believe in a different approach: **awareness and dialogue**.  
- This project is built as **open source**, so anyone can join, experiment, and improve it together.  

---

## 3. Current Status
- Core pipeline: **BERT + AutoEncoder + SHAP** anomaly detection.  
- Outputs JSON with detected “risk candidates.”  
- Known limitations:  
  - Low precision/recall  
  - Limited dataset (tested with small browser history samples)  
  - UX not ready for non-developers  

---

## 4. Getting Started

### Install
```bash
git clone https://github.com/tacsotai/file2risks.git
cd file2risks
pip install -r requirements.txt
```

### Example Run
```bash
python main.py ~/Downloads/takeout-yyyyMM..XXX.zip
```

### Example Output
```json
{
    "results": [
        {
            "reason": {
                "hoge": 0.644,
                "foo": 0.3,
                "bar": 0.055
            },
            "title": "Potential risk detected on example.com",
            "content": "Browsing suspicious patterns...",
            "timing_at": "2025-08-10"
        }
    ]
}
```

---

## 5. How to Contribute
- **Developers** → improve anomaly detection, experiment with models (PyOD, PyCaret, transformers, etc.)  
- **Designers / UX** → propose how insights should be visualized for parents/kids  
- **Parents / Users** → share scenarios, feedback, or small datasets (if comfortable)  

👉 Contributions of any size are welcome: fixing typos, raising issues, writing docs, or submitting PRs.  

---

## 6. Roadmap
- [ ] Better README (with diagrams, examples)  
- [ ] Support more data sources (chat history, messaging logs)  
- [ ] Compare anomaly detection methods beyond BERT-AE  
- [ ] Integrate with [RiskAware App](https://riskaware.sotai.co)
- [ ] Build a small community (Discord/Slack)  

---

## 7. License
[MIT](./LICENSE) (open for personal and commercial use, attribution appreciated)  

---

## 8. Community
- Coming soon: Discord/Slack for discussions  
- For now: please use GitHub **Issues** & **Discussions**  
