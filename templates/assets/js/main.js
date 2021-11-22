import "/assets/js/jquery-csv.js";
import * as $ from "jquery";


// var csv = require('jquery-csv');
var isCelcius = true;
var unitsBtn = document.getElementById("units-btn");
var unitsBtnImg = document.getElementById("units-img");
var cityCBox = document.getElementById("cityDropDown");
var celciusIcon = "https://cdn3.iconfinder.com/data/icons/aami-web-internet/64/aami9-50-128.png";
var farenheitIcon = "https://cdn3.iconfinder.com/data/icons/aami-web-internet/64/aami9-49-128.png";
var selectedCity = "";
var cityData = $.csv.toArrays("/templates/assets/TempData/World Cities Lat Lon (Small).csv");

var x;
for(x = 0; x < 10; x++){
  console.log(cityData[x]);
}

function swapTempUnits(){
  if(isCelcius){
    isCelcius = false;
    unitsBtnImg.src = farenheitIcon;
  }
  else{
    isCelcius = true;
    unitsBtnImg.src = celciusIcon;
  }
}

function citySelected(){
  var selectedCity = cityCBox.value;
  let latitude = cityData[selectedCity][0];
}
