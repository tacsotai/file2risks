# file2risks

## 1. Project Overview
**file2risks** is an open-source module designed to analyze browser history (and other digital traces) to detect **potential risks** and uncover **opportunities** for safer digital habits.  
It is part of the larger [RiskAware](https://www.sotai.co/) project, which empowers families and individuals to use technology more safely and effectively.  

⚠️ **Early stage**: Results are still in development, and predictions may not yet be fully accurate. Your contributions can help improve this tool for everyone!  

---

## 2. Motivation & Vision
- Parents and individuals often face the challenge of balancing **freedom and safety** in the digital world.  
- Existing tools focus heavily on **blocking or monitoring**, which can feel restrictive.  
- We believe in a more empowering approach: **awareness and informed decision-making**.  
- By making this project **open source**, we invite a global community to collaborate, innovate, and refine the system together.  

---

## 3. Current Status
- Core pipeline: **BERT + AutoEncoder + SHAP** anomaly detection.  
- Outputs JSON with detected “risk candidates” and potential insights.  
- Known limitations:  
  - Low precision/recall due to limited training data.  
  - Small dataset (tested with minimal browser history samples).  
  - User experience is still developer-focused and not yet optimized for general users.  

---

## 4. How It Works

1. **Upload Your Data**  
   Provide your browser history or other digital traces to the system.  
2. **Analysis in Progress**  
   The system processes your data using advanced AI models.  
3. **Receive Notifications**  
   Once the analysis is complete, you’ll be notified and can review the results, including potential risks and actionable insights.

---

## 5. Getting Started

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

## 6. How to Contribute
- **Developers** → Enhance anomaly detection, experiment with models (e.g., PyOD, PyCaret, transformers).  
- **Designers / UX** → Propose user-friendly ways to visualize insights for families and individuals.  
- **Parents / Users** → Share real-world scenarios, feedback, or small datasets (if comfortable).  

👉 Contributions of any size are welcome: fixing typos, raising issues, writing documentation, or submitting PRs.  

---

## 7. Roadmap
- [ ] Improve README with diagrams and examples.  
- [ ] Expand support to include more data sources (e.g., chat history, messaging logs).  
- [ ] Compare anomaly detection methods beyond BERT-AE.  
- [ ] Integrate with the [RiskAware App](https://riskaware.sotai.co).  
- [ ] Build a small community (Discord/Slack) for collaboration.  

---

## 8. License
[MIT](./LICENSE) (open for personal and commercial use, attribution appreciated).  

---

## 9. Community
- Coming soon: Discord/Slack for discussions.  
- For now: Please use GitHub **Issues** & **Discussions** for feedback and collaboration.  

---

## 10. Donation

If you find **file2risks** valuable and want to support its development,  
you can donate Bitcoin to the following address:

**Bitcoin Address (BTC):**

```
3LvyGJFgzP5ox3X4sLYun4ojz3xuYYcPbk
```
