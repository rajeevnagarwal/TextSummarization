var express = require('express');
var router = express.Router();
var pythonShell = require('python-shell');
var scriptPath = './scripts/';
var script = 'summarization.py';
var bodyParser = require('body-parser');

router.use(bodyParser.urlencoded());
router.use(bodyParser.json());
router.route('/')
.get(function(req,res){
	
})
.post(function(req,res)
{
  var arg = req.body.text;
  var amt = req.body.amt;
  var options = {
  mode: 'text',
  pythonOptions: ['-u'],
  scriptPath:scriptPath,
  args: [arg,amt]
};
	pythonShell.run(script,options,function(err,results){
		if(err) throw err;
		var result = results[0];
		result.trim();
		res.render('index',{summary:result,text:arg});
		/*console.log(res);*/
		
	});
	
});

module.exports = router;

