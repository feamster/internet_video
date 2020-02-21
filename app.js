var express = require("express");
var app = express();
 
app.use(express.static("controllers")).listen(8080);