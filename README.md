Home Depot
===

Analyzing Home Depot search querying data to improve search results

Dependencies
---

- `Python 3.9`
- `spaCy 3.3`
  * `en-core-web-lg`
- `pandas 1.4.2`
- `numpy 1.22.4`
- `scipy 1.8.1`
- `scikit-learn 1.1.1`
- `matplotlib 3.5.2`
- `seaborn 0.11.2`

Setup
---

```
$ git clone git@github.com:Josef-Hlink/Home-Depot.git   # assuming SSH
$ cd Home-Depot
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python3 -m spacy download en_core_web_lg       # if it does not install from requirements
(venv) $ cd src
(venv) $ python3 main.py -h
```

If `python3` doesn't work, try `python`

The `-h` flag is to show some help on usage of the script
