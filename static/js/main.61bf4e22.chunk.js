(window.webpackJsonp=window.webpackJsonp||[]).push([[0],{26:function(e,n,t){e.exports=t(47)},41:function(e,n,t){},47:function(e,n,t){"use strict";t.r(n);var o,r=t(1),i=t.n(r),a=t(13),c=t.n(a),s=t(23),l=t(49),u=(t(35),t(37),t(39),t(41),t(4)),d=t(9),f=(t(16),t(17),t(18)),w=t(19),h=t(5),p=Object(h.fromJS)({}),v=Object(w.createReducer)(p,{}),g=Object(f.combineReducers)({app:v});o=Object(u.applyMiddleware)(d.a);var b=Object(u.createStore)(g,o),m=t(20),j=t(21),k=t(24),O=t(22),y=t(25),E=function(e){function n(){return Object(m.a)(this,n),Object(k.a)(this,Object(O.a)(n).apply(this,arguments))}return Object(y.a)(n,e),Object(j.a)(n,[{key:"render",value:function(){return i.a.createElement("div",{className:"app"})}}]),n}(r.Component),W=Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));function A(e,n){navigator.serviceWorker.register(e).then(function(e){e.onupdatefound=function(){var t=e.installing;t.onstatechange=function(){"installed"===t.state&&(navigator.serviceWorker.controller?(console.log("New content is available and will be used when all tabs for this page are closed. See http://bit.ly/CRA-PWA."),n&&n.onUpdate&&n.onUpdate(e)):(console.log("Content is cached for offline use."),n&&n.onSuccess&&n.onSuccess(e)))}}}).catch(function(e){console.error("Error during service worker registration:",e)})}var R=document.getElementById("root");Promise.resolve(b).then(function(e){c.a.render(i.a.createElement(s.a,{store:e},i.a.createElement(l.a,null,i.a.createElement(E,null))),R)}),function(e){if("serviceWorker"in navigator){if(new URL("",window.location).origin!==window.location.origin)return;window.addEventListener("load",function(){var n="".concat("","/service-worker.js");W?(function(e,n){fetch(e).then(function(t){404===t.status||-1===t.headers.get("content-type").indexOf("javascript")?navigator.serviceWorker.ready.then(function(e){e.unregister().then(function(){window.location.reload()})}):A(e,n)}).catch(function(){console.log("No internet connection found. App is running in offline mode.")})}(n,e),navigator.serviceWorker.ready.then(function(){console.log("This web app is being served cache-first by a service worker. To learn more, visit http://bit.ly/CRA-PWA")})):A(n,e)})}}()}},[[26,2,1]]]);
//# sourceMappingURL=main.61bf4e22.chunk.js.map