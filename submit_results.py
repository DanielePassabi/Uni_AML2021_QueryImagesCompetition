import requests
import json

def submit(results, url):
    res = json.dumps(results)
    response = requests.post(url, res)
    result = json.loads(response.text)
    print(f"accuracy is {result['results']}")


url = "http://kamino.disi.unitn.it:3001/results/"

mydata = dict()
mydata['groupname'] = "the wardogs"

res = dict()
# res['<query image name>'] = ['<gallery image rank 1>', '<gallery image rank 2>', ..., '<gallery image rank 10>']
res["img001.jpg"] = ["gal999.jpg", "gal345.jpg", "gal268.jpg", "gal180.jpg", "gal008.jpg", "gal316.jpg", "gal423.jpg", "gal111.jpg", "gal234.jpg", "gal730.jpg"]
res["img002.jpg"] = ["gal336.jpg", "gal422.jpg", "gal194.jpg", "gal644.jpg", "gal910.jpg", "gal108.jpg", "gal179.jpg", "gal873.jpg", "gal556.jpg", "gal692.jpg"]
res["img003.jpg"] = ["gal098.jpg", "gal879.jpg", "gal883.jpg", "gal556.jpg", "gal642.jpg", "gal329.jpg", "gal305.jpg", "gal068.jpg", "gal080.jpg", "gal018.jpg"]
mydata["images"] = res
submit(mydata, url)


