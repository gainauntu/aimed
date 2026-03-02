import argparse,uuid,json,requests
ap=argparse.ArgumentParser(); ap.add_argument('--url',default='http://127.0.0.1:9000'); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); ap.add_argument('--timeout',type=int,default=30); args=ap.parse_args();
req={'request_id':str(uuid.uuid4()),'side_a_path':args.a,'side_b_path':args.b};
r=requests.post(args.url.rstrip('/')+'/predict',json=req,timeout=(5,args.timeout));
print('HTTP',r.status_code); print(json.dumps(r.json(),ensure_ascii=False,indent=2));
