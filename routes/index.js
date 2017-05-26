var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index',{ summary:"Summary of the text will be generated here",text:"Enter your text"});
});

module.exports = router;
