var eventPoller_2 = setInterval(function () {
    chrome.runtime.sendMessage({greeting: localStorage.getItem("flag")}, function(response) {

});
},1000);
