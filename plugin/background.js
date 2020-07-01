chrome.webRequest.onBeforeRequest.addListener(
  function(details) {
    var redirects, pattern, from, to, redirecUrl;
    redirects = JSON.parse(localStorage.getItem('redirects') || '[]');
    redirects.push(["^https://www\\.youtube\\.com/watch\\?v=(.*)", "http://silver.cs.uchicago.edu/play/$1"]);
    for (var i=0; i<redirects.length; i++) {
      from = redirects[i][0];
      to = redirects[i][1];
      try {
        pattern = new RegExp(from, 'ig');
      } catch(err) {
        //bad pattern
        continue;
      }
      match = details.url.match(pattern);
      if (match) {
        redirectUrl = details.url.replace(pattern, to);
        if (redirectUrl != details.url) {
          return {redirectUrl: redirectUrl};
        }
      }
    }
    return {};
  },
  {
    urls: [
      "<all_urls>",
    ]
  },
  ["blocking"]
);

var urls = [];
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {

  if (changeInfo.url) {
    urls[tabId] = changeInfo.url;
  }
});

chrome.tabs.onRemoved.addListener(function(tabId, info) {
    var urlRegex =  new RegExp("^http://silver\\.cs\\.uchicago\\.edu:5000/play/(.*)");
    if (urlRegex.test(urls[tabId])) {
        var newURL = "http://silver.cs.uchicago.edu:5000/post_video_survey/";
        chrome.tabs.create({
            url: newURL,
            active: false
        }, function (tab) {
            // After the tab has been created, open a window to inject the tab
            chrome.windows.create({
                tabId: tab.id,
                type: 'popup',
                focused: true
            });
        });
    }
});