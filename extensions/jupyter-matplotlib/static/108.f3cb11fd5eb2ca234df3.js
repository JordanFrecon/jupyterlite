(self.webpackChunkjupyter_matplotlib=self.webpackChunkjupyter_matplotlib||[]).push([[108,439],{686:(t,e,i)=>{"use strict";i.r(e),i.d(e,{MODULE_NAME:()=>l.o,MODULE_VERSION:()=>l.Y,MPLCanvasModel:()=>c,MPLCanvasView:()=>_,ToolbarModel:()=>u,ToolbarView:()=>p});var s=i(439),o=i(395);function a(t,e){const i=e.getBoundingClientRect();return{x:t.clientX-i.left,y:t.clientY-i.top}}function n(t){return Object.keys(t).reduce(((e,i)=>("object"!=typeof t[i]&&(e[i]=t[i]),e)),{})}function r(t){const e=t.getContext("2d");if(null===e)throw"Could not create 2d context.";return e}function h(t){const e=[];return t.ctrlKey&&e.push("ctrl"),t.altKey&&e.push("alt"),t.shiftKey&&e.push("shift"),t.metaKey&&e.push("meta"),e}var l=i(657),d=function(t,e,i,s){return new(i||(i=Promise))((function(o,a){function n(t){try{h(s.next(t))}catch(t){a(t)}}function r(t){try{h(s.throw(t))}catch(t){a(t)}}function h(t){var e;t.done?o(t.value):(e=t.value,e instanceof i?e:new i((function(t){t(e)}))).then(n,r)}h((s=s.apply(t,e||[])).next())}))};class c extends o.DOMWidgetModel{defaults(){return Object.assign(Object.assign({},super.defaults()),{_model_name:"MPLCanvasModel",_view_name:"MPLCanvasView",_model_module:"jupyter-matplotlib",_view_module:"jupyter-matplotlib",_model_module_version:"^"+l.Y,_view_module_version:"^"+l.Y,header_visible:!0,footer_visible:!0,toolbar:null,toolbar_visible:"fade-in-fade-out",toolbar_position:"horizontal",resizable:!0,capture_scroll:!1,pan_zoom_throttle:33,_data_url:null,_size:[0,0],_figure_label:"Figure",_message:"",_cursor:"pointer",_image_mode:"full",_rubberband_x:0,_rubberband_y:0,_rubberband_width:0,_rubberband_height:0})}initialize(t,e){super.initialize(t,e),this.offscreen_canvas=document.createElement("canvas"),this.offscreen_context=r(this.offscreen_canvas);const i=this.offscreen_context,s=i.backingStorePixelRatio||i.webkitBackingStorePixelRatio||i.mozBackingStorePixelRatio||i.msBackingStorePixelRatio||i.oBackingStorePixelRatio||1;this.requested_size=null,this.resize_requested=!1,this.ratio=(window.devicePixelRatio||1)/s,this.resize_canvas(),this._init_image(),this.on("msg:custom",this.on_comm_message.bind(this)),this.on("change:resizable",(()=>{this._for_each_view((t=>{t.update_canvas()}))})),this.on("change:_size",(()=>{this.resize_canvas(),this.offscreen_context.drawImage(this.image,0,0)})),this.on("comm_live_update",this.update_disabled.bind(this)),this.update_disabled(),this.send_initialization_message()}get size(){return this.get("_size")}get disabled(){return!this.comm_live}update_disabled(){this.set("resizable",!this.disabled)}sync(t,e,i={}){i.attrs&&delete i.attrs._data_url,super.sync(t,e,i)}send_message(t,e={}){e.type=t,this.send(e,{})}send_initialization_message(){1!==this.ratio&&(this.send_message("set_dpi_ratio",{dpi_ratio:this.ratio}),this.send_message("set_device_pixel_ratio",{device_pixel_ratio:this.ratio})),this.send_message("refresh"),this.send_message("send_image_mode"),this.send_message("initialized")}send_draw_message(){this.waiting_for_image||(this.waiting_for_image=!0,this.send_message("draw"))}handle_save(){const t=document.createElement("a");t.href=this.offscreen_canvas.toDataURL(),t.download=this.get("_figure_label")+".png",document.body.appendChild(t),t.click(),document.body.removeChild(t)}handle_resize(t){this.resize_canvas(),this.offscreen_context.drawImage(this.image,0,0),this.resize_requested||this._for_each_view((t=>{t.resize_and_update_canvas(this.size)})),this.send_message("refresh"),this.resize_requested=!1,null!==this.requested_size&&(this.resize(this.requested_size[0],this.requested_size[1]),this.requested_size=null)}resize(t,e){t<=5||e<=5||(this._for_each_view((i=>{i.resize_and_update_canvas([t,e])})),this.resize_requested?this.requested_size=[t,e]:(this.resize_requested=!0,this.send_message("resize",{width:t,height:e})))}resize_canvas(){this.offscreen_canvas.width=this.size[0]*this.ratio,this.offscreen_canvas.height=this.size[1]*this.ratio}handle_rubberband(t){let e=t.x0/this.ratio,i=(this.offscreen_canvas.height-t.y0)/this.ratio,s=t.x1/this.ratio,o=(this.offscreen_canvas.height-t.y1)/this.ratio;e=Math.floor(e)+.5,i=Math.floor(i)+.5,s=Math.floor(s)+.5,o=Math.floor(o)+.5,this.set("_rubberband_x",Math.min(e,s)),this.set("_rubberband_y",Math.min(i,o)),this.set("_rubberband_width",Math.abs(s-e)),this.set("_rubberband_height",Math.abs(o-i)),this.save_changes(),this._for_each_view((t=>{t.update_canvas()}))}handle_draw(t){this.send_draw_message()}handle_binary(t,e){const i=window.URL||window.webkitURL,s=new Uint8Array(ArrayBuffer.isView(e[0])?e[0].buffer:e[0]),o=new Blob([s],{type:"image/png"}),a=i.createObjectURL(o);this.image.src&&i.revokeObjectURL(this.image.src),this.image.src=a,this.set("_data_url",this.offscreen_canvas.toDataURL()),this.waiting_for_image=!1}handle_history_buttons(t){}handle_navigate_mode(t){}on_comm_message(t,e){const i=JSON.parse(t.data),s=i.type;let o;try{o=this["handle_"+s].bind(this)}catch(t){return void console.log("No handler for the '"+s+"' message type: ",i)}o&&o(i,e)}_init_image(){this.image=new Image,this.image.onload=()=>{if(this.disabled)return this.offscreen_canvas.width=this.image.width,this.offscreen_canvas.height=this.image.height,this.offscreen_context.drawImage(this.image,0,0),void this._for_each_view((t=>{t.canvas.width=this.image.width/this.ratio,t.canvas.height=this.image.height/this.ratio,t.canvas.style.width=t.canvas.width+"px",t.canvas.style.height=t.canvas.height+"px",t.top_canvas.width=this.image.width/this.ratio,t.top_canvas.height=this.image.height/this.ratio,t.top_canvas.style.width=t.top_canvas.width+"px",t.top_canvas.style.height=t.top_canvas.height+"px",t.canvas_div.style.width=t.canvas.width+"px",t.canvas_div.style.height=t.canvas.height+"px",t.update_canvas(!0)}));"full"===this.get("_image_mode")&&this.offscreen_context.clearRect(0,0,this.offscreen_canvas.width,this.offscreen_canvas.height),this.offscreen_context.drawImage(this.image,0,0),this._for_each_view((t=>{t.update_canvas()}))};const t=this.get("_data_url");null!==t&&(this.image.src=t)}_for_each_view(t){for(const e in this.views)this.views[e].then((e=>{t(e)}))}remove(){this.send_message("closing")}}c.serializers=Object.assign(Object.assign({},o.DOMWidgetModel.serializers),{toolbar:{deserialize:o.unpack_models}});class _ extends o.DOMWidgetView{render(){return d(this,void 0,void 0,(function*(){this.resizing=!1,this.resize_handle_size=20,this.el.classList.add("jupyter-matplotlib"),this.figure=document.createElement("div"),this.figure.classList.add("jupyter-matplotlib-figure"),this.el.appendChild(this.figure),this._init_header(),this._init_canvas(),yield this._init_toolbar(),this._init_footer(),this._resize_event=this.resize_event.bind(this),this._stop_resize_event=this.stop_resize_event.bind(this),window.addEventListener("mousemove",this._resize_event),window.addEventListener("mouseup",this._stop_resize_event),this.el.addEventListener("mouseenter",(()=>{this.toolbar_view.fade_in()})),this.el.addEventListener("mouseleave",(()=>{this.toolbar_view.fade_out()})),this.model_events()}))}model_events(){this.model.on("change:header_visible",this._update_header_visible.bind(this)),this.model.on("change:footer_visible",this._update_footer_visible.bind(this)),this.model.on("change:toolbar_visible",this._update_toolbar_visible.bind(this)),this.model.on("change:toolbar_position",this._update_toolbar_position.bind(this)),this.model.on("change:_figure_label",this._update_figure_label.bind(this)),this.model.on("change:_message",this._update_message.bind(this)),this.model.on("change:_cursor",this._update_cursor.bind(this))}_update_header_visible(){this.header.style.display=this.model.get("header_visible")?"":"none"}_update_footer_visible(){this.footer.style.display=this.model.get("footer_visible")?"":"none"}_update_toolbar_visible(){this.toolbar_view.set_visibility(this.model.get("toolbar_visible"))}_update_toolbar_position(){this.model.get("toolbar").set("position",this.model.get("toolbar_position"))}_init_header(){this.header=document.createElement("div"),this.header.classList.add("jupyter-widgets","widget-label","jupyter-matplotlib-header"),this._update_header_visible(),this._update_figure_label(),this.figure.appendChild(this.header)}_update_figure_label(){this.header.textContent=this.model.get("_figure_label")}_init_canvas(){const t=document.createElement("div");t.classList.add("jupyter-widgets","jupyter-matplotlib-canvas-container"),this.figure.appendChild(t);const e=this.canvas_div=document.createElement("div");e.style.position="relative",e.style.clear="both",e.classList.add("jupyter-widgets","jupyter-matplotlib-canvas-div"),e.addEventListener("keydown",this.key_event("key_press")),e.addEventListener("keyup",this.key_event("key_release")),e.setAttribute("tabindex","0"),t.appendChild(e);const i=this.canvas=document.createElement("canvas");i.style.display="block",i.style.position="absolute",i.style.left="0",i.style.top="0",i.style.zIndex="0",this.context=r(i);const o=this.top_canvas=document.createElement("canvas");o.style.display="block",o.style.position="absolute",o.style.left="0",o.style.top="0",o.style.zIndex="1",o.addEventListener("dblclick",this.mouse_event("dblclick")),o.addEventListener("mousedown",this.mouse_event("button_press")),o.addEventListener("mouseup",this.mouse_event("button_release")),o.addEventListener("mousemove",(0,s.throttle)(this.mouse_event("motion_notify"),this.model.get("pan_zoom_throttle"))),o.addEventListener("mouseenter",this.mouse_event("figure_enter")),o.addEventListener("mouseleave",this.mouse_event("figure_leave")),o.addEventListener("wheel",(0,s.throttle)(this.mouse_event("scroll"),this.model.get("pan_zoom_throttle"))),o.addEventListener("wheel",(t=>{this.model.get("capture_scroll")&&t.preventDefault()})),e.appendChild(i),e.appendChild(o),this.top_context=r(o),this.top_context.strokeStyle="rgba(0, 0, 0, 255)",this.top_canvas.addEventListener("contextmenu",(t=>(t.preventDefault(),t.stopPropagation(),!1))),this.resize_and_update_canvas(this.model.size)}_init_toolbar(){return d(this,void 0,void 0,(function*(){this.toolbar_view=yield this.create_child_view(this.model.get("toolbar")),this.figure.appendChild(this.toolbar_view.el),this._update_toolbar_position(),this._update_toolbar_visible()}))}update_canvas(t=!1){if(0!==this.canvas.width&&0!==this.canvas.height){if(this.top_context.save(),this.context.clearRect(0,0,this.canvas.width,this.canvas.height),t?this.context.drawImage(this.model.offscreen_canvas,0,0,this.canvas.width,this.canvas.height):this.context.drawImage(this.model.offscreen_canvas,0,0),this.top_context.clearRect(0,0,this.top_canvas.width,this.top_canvas.height),0!==this.model.get("_rubberband_width")&&0!==this.model.get("_rubberband_height")&&(this.top_context.strokeStyle="gray",this.top_context.lineWidth=1,this.top_context.shadowColor="black",this.top_context.shadowBlur=2,this.top_context.shadowOffsetX=1,this.top_context.shadowOffsetY=1,this.top_context.strokeRect(this.model.get("_rubberband_x"),this.model.get("_rubberband_y"),this.model.get("_rubberband_width"),this.model.get("_rubberband_height"))),this.model.get("resizable")){const t=this.top_context.createLinearGradient(this.top_canvas.width-this.resize_handle_size,this.top_canvas.height-this.resize_handle_size,this.top_canvas.width,this.top_canvas.height);t.addColorStop(0,"white"),t.addColorStop(1,"black"),this.top_context.fillStyle=t,this.top_context.strokeStyle="gray",this.top_context.globalAlpha=.3,this.top_context.beginPath(),this.top_context.moveTo(this.top_canvas.width,this.top_canvas.height),this.top_context.lineTo(this.top_canvas.width,this.top_canvas.height-this.resize_handle_size),this.top_context.lineTo(this.top_canvas.width-this.resize_handle_size,this.top_canvas.height),this.top_context.closePath(),this.top_context.fill(),this.top_context.stroke()}this.top_context.restore()}}_update_cursor(){this.top_canvas.style.cursor=this.model.get("_cursor")}_init_footer(){this.footer=document.createElement("div"),this.footer.classList.add("jupyter-widgets","widget-label","jupyter-matplotlib-footer"),this._update_footer_visible(),this._update_message(),this.figure.appendChild(this.footer)}_update_message(){this.footer.textContent=this.model.get("_message")}resize_and_update_canvas(t){this.canvas.setAttribute("width",""+t[0]*this.model.ratio),this.canvas.setAttribute("height",""+t[1]*this.model.ratio),this.canvas.style.width=t[0]+"px",this.canvas.style.height=t[1]+"px",this.top_canvas.setAttribute("width",String(t[0])),this.top_canvas.setAttribute("height",String(t[1])),this.canvas_div.style.width=t[0]+"px",this.canvas_div.style.height=t[1]+"px",this.update_canvas()}mouse_event(t){return e=>{const i=a(e,this.top_canvas);if("scroll"===t&&(e.data="scroll",e.deltaY<0?e.step=1:e.step=-1),"button_press"===t){if(i.x>=this.top_canvas.width-this.resize_handle_size&&i.y>=this.top_canvas.height-this.resize_handle_size&&this.model.get("resizable"))return void(this.resizing=!0);this.canvas.focus(),this.canvas_div.focus()}if(this.resizing)return;"motion_notify"===t&&(i.x>=this.top_canvas.width-this.resize_handle_size&&i.y>=this.top_canvas.height-this.resize_handle_size?this.top_canvas.style.cursor="nw-resize":this.top_canvas.style.cursor=this.model.get("_cursor"));const s=i.x*this.model.ratio,o=i.y*this.model.ratio;this.model.send_message(t,{x:s,y:o,button:e.button,step:e.step,modifiers:h(e),guiEvent:n(e)})}}resize_event(t){if(this.resizing){const e=a(t,this.top_canvas);this.model.resize(e.x,e.y)}}stop_resize_event(){this.resizing=!1}key_event(t){return e=>{if(e.stopPropagation(),e.preventDefault(),"key_press"===t){if(e.key===this._key)return;this._key=e.key}"key_release"===t&&(this._key=null);let i="";return e.ctrlKey&&"Control"!==e.key?i+="ctrl+":e.altKey&&"Alt"!==e.key?i+="alt+":e.shiftKey&&"Shift"!==e.key&&(i+="shift+"),i+="k"+e.key,this.model.send_message(t,{key:i,guiEvent:n(e)}),!1}}remove(){window.removeEventListener("mousemove",this._resize_event),window.removeEventListener("mouseup",this._stop_resize_event)}}i(351);class u extends o.DOMWidgetModel{defaults(){return Object.assign(Object.assign({},super.defaults()),{_model_name:"ToolbarModel",_view_name:"ToolbarView",_model_module:"jupyter-matplotlib",_view_module:"jupyter-matplotlib",_model_module_version:"^"+l.Y,_view_module_version:"^"+l.Y,toolitems:[],position:"left",button_style:"",_current_action:""})}}class p extends o.DOMWidgetView{constructor(){super(...arguments),this.visibility="fade-in-fade-out"}initialize(t){super.initialize(t),this.on("comm_live_update",this.update_disabled.bind(this))}render(){this.el.classList.add("jupyter-widgets","jupyter-matplotlib-toolbar","widget-container","widget-box"),this.set_visibility("fade-in-fade-out"),this.create_toolbar(),this.model_events()}create_toolbar(){const t=this.model.get("toolitems");this.toolbar=document.createElement("div"),this.toolbar.classList.add("widget-container","widget-box"),this.el.appendChild(this.toolbar),this.buttons={};for(const e in t){const i=t[e][0],s=t[e][1],o=t[e][2],a=t[e][3];if(!i)continue;const n=document.createElement("button");n.classList.add("jupyter-matplotlib-button","jupyter-widgets","jupyter-button"),n.setAttribute("href","#"),n.setAttribute("title",s),n.style.outline="none",n.addEventListener("click",this.toolbar_button_onclick(a));const r=document.createElement("i");r.classList.add("center","fa","fa-fw","fa-"+o),n.appendChild(r),this.buttons[a]=n,this.toolbar.appendChild(n)}this.set_position(),this.set_buttons_style(),this.update_disabled()}get disabled(){return!this.model.comm_live}update_disabled(){this.disabled&&(this.toolbar.style.display="none")}set_position(){const t=this.model.get("position");"left"===t||"right"===t?(this.el.classList.remove("widget-hbox"),this.el.classList.add("widget-vbox"),this.toolbar.classList.remove("widget-hbox"),this.toolbar.classList.add("widget-vbox"),this.el.style.top="3px",this.el.style.bottom="auto","left"===t?(this.el.style.left="3px",this.el.style.right="auto"):(this.el.style.left="auto",this.el.style.right="3px")):(this.el.classList.add("widget-hbox"),this.el.classList.remove("widget-vbox"),this.toolbar.classList.add("widget-hbox"),this.toolbar.classList.remove("widget-vbox"),this.el.style.right="3px",this.el.style.left="auto","top"===t?(this.el.style.top="3px",this.el.style.bottom="auto"):(this.el.style.top="auto",this.el.style.bottom="3px"))}toolbar_button_onclick(t){return e=>{"pan"!==t&&"zoom"!==t||(this.model.get("_current_action")===t?this.model.set("_current_action",""):this.model.set("_current_action",t),this.model.save_changes()),this.send({type:"toolbar_button",name:t})}}set_visibility(t){return"boolean"==typeof t&&(t=t?"visible":"hidden"),this.visibility=t,"fade-in-fade-out"===t?(this.el.classList.add("jupyter-matplotlib-toolbar-fadein-fadeout"),this.el.style.visibility="hidden",void(this.el.style.opacity="0")):(this.el.classList.remove("jupyter-matplotlib-toolbar-fadein-fadeout"),"visible"===t?(this.el.style.visibility="visible",void(this.el.style.opacity="1")):(this.el.style.visibility="hidden",void(this.el.style.opacity="0")))}fade_in(){"fade-in-fade-out"===this.visibility&&(this.el.style.visibility="visible",this.el.style.opacity="1")}fade_out(){"fade-in-fade-out"===this.visibility&&(this.el.style.visibility="hidden",this.el.style.opacity="0")}set_buttons_style(){const t={primary:["mod-primary"],success:["mod-success"],info:["mod-info"],warning:["mod-warning"],danger:["mod-danger"]};for(const e in this.buttons){const i=this.buttons[e];for(const e in t)i.classList.remove(t[e]);i.classList.remove("mod-active");const s=this.model.get("button_style");""!==s&&i.classList.add(t[s]),e===this.model.get("_current_action")&&i.classList.add("mod-active")}}model_events(){this.model.on("change:position",this.set_position.bind(this)),this.model.on_some_change(["button_style","_current_action"],this.set_buttons_style,this)}}},153:(t,e,i)=>{(e=i(645)(!1)).push([t.id,".jupyter-matplotlib {\n    width: auto;\n    height: auto;\n    flex: 1 1 auto;\n}\n\n/* Toolbar */\n\n.jupyter-matplotlib-toolbar {\n    position: absolute;\n    overflow: visible;\n    z-index: 100;\n}\n\n.jupyter-matplotlib-toolbar-fadein-fadeout {\n    transition: visibility 0.5s linear, opacity 0.5s linear;\n}\n\n.jupyter-matplotlib-button {\n    width: calc(var(--jp-widgets-inline-width-tiny) / 2 - 2px);\n    padding: 0 !important;\n}\n\n/* Figure */\n\n.jupyter-matplotlib-figure {\n    width: fit-content;\n    height: auto;\n    overflow: hidden;\n    display: flex;\n    flex-direction: column;\n}\n\n.jupyter-matplotlib-canvas-container {\n    overflow: auto;\n}\n\n.jupyter-matplotlib-canvas-div {\n    margin: 2px;\n    flex: 1 1 auto;\n}\n\n.jupyter-matplotlib-header {\n    width: 100%;\n    text-align: center;\n}\n\n.jupyter-matplotlib-footer {\n    width: 100%;\n    text-align: center;\n    min-height: var(--jp-widgets-inline-height);\n}\n",""]),t.exports=e},645:t=>{"use strict";t.exports=function(t){var e=[];return e.toString=function(){return this.map((function(e){var i=function(t,e){var i,s,o,a=t[1]||"",n=t[3];if(!n)return a;if(e&&"function"==typeof btoa){var r=(i=n,s=btoa(unescape(encodeURIComponent(JSON.stringify(i)))),o="sourceMappingURL=data:application/json;charset=utf-8;base64,".concat(s),"/*# ".concat(o," */")),h=n.sources.map((function(t){return"/*# sourceURL=".concat(n.sourceRoot||"").concat(t," */")}));return[a].concat(h).concat([r]).join("\n")}return[a].join("\n")}(e,t);return e[2]?"@media ".concat(e[2]," {").concat(i,"}"):i})).join("")},e.i=function(t,i,s){"string"==typeof t&&(t=[[null,t,""]]);var o={};if(s)for(var a=0;a<this.length;a++){var n=this[a][0];null!=n&&(o[n]=!0)}for(var r=0;r<t.length;r++){var h=[].concat(t[r]);s&&o[h[0]]||(i&&(h[2]?h[2]="".concat(i," and ").concat(h[2]):h[2]=i),e.push(h))}},e}},351:(t,e,i)=>{var s=i(379),o=i(153);"string"==typeof(o=o.__esModule?o.default:o)&&(o=[[t.id,o,""]]);s(o,{insert:"head",singleton:!1}),t.exports=o.locals||{}},379:(t,e,i)=>{"use strict";var s,o=function(){var t={};return function(e){if(void 0===t[e]){var i=document.querySelector(e);if(window.HTMLIFrameElement&&i instanceof window.HTMLIFrameElement)try{i=i.contentDocument.head}catch(t){i=null}t[e]=i}return t[e]}}(),a=[];function n(t){for(var e=-1,i=0;i<a.length;i++)if(a[i].identifier===t){e=i;break}return e}function r(t,e){for(var i={},s=[],o=0;o<t.length;o++){var r=t[o],h=e.base?r[0]+e.base:r[0],l=i[h]||0,d="".concat(h," ").concat(l);i[h]=l+1;var c=n(d),_={css:r[1],media:r[2],sourceMap:r[3]};-1!==c?(a[c].references++,a[c].updater(_)):a.push({identifier:d,updater:v(_,e),references:1}),s.push(d)}return s}function h(t){var e=document.createElement("style"),s=t.attributes||{};if(void 0===s.nonce){var a=i.nc;a&&(s.nonce=a)}if(Object.keys(s).forEach((function(t){e.setAttribute(t,s[t])})),"function"==typeof t.insert)t.insert(e);else{var n=o(t.insert||"head");if(!n)throw new Error("Couldn't find a style target. This probably means that the value for the 'insert' parameter is invalid.");n.appendChild(e)}return e}var l,d=(l=[],function(t,e){return l[t]=e,l.filter(Boolean).join("\n")});function c(t,e,i,s){var o=i?"":s.media?"@media ".concat(s.media," {").concat(s.css,"}"):s.css;if(t.styleSheet)t.styleSheet.cssText=d(e,o);else{var a=document.createTextNode(o),n=t.childNodes;n[e]&&t.removeChild(n[e]),n.length?t.insertBefore(a,n[e]):t.appendChild(a)}}function _(t,e,i){var s=i.css,o=i.media,a=i.sourceMap;if(o?t.setAttribute("media",o):t.removeAttribute("media"),a&&"undefined"!=typeof btoa&&(s+="\n/*# sourceMappingURL=data:application/json;base64,".concat(btoa(unescape(encodeURIComponent(JSON.stringify(a))))," */")),t.styleSheet)t.styleSheet.cssText=s;else{for(;t.firstChild;)t.removeChild(t.firstChild);t.appendChild(document.createTextNode(s))}}var u=null,p=0;function v(t,e){var i,s,o;if(e.singleton){var a=p++;i=u||(u=h(e)),s=c.bind(null,i,a,!1),o=c.bind(null,i,a,!0)}else i=h(e),s=_.bind(null,i,e),o=function(){!function(t){if(null===t.parentNode)return!1;t.parentNode.removeChild(t)}(i)};return s(t),function(e){if(e){if(e.css===t.css&&e.media===t.media&&e.sourceMap===t.sourceMap)return;s(t=e)}else o()}}t.exports=function(t,e){(e=e||{}).singleton||"boolean"==typeof e.singleton||(e.singleton=(void 0===s&&(s=Boolean(window&&document&&document.all&&!window.atob)),s));var i=r(t=t||[],e);return function(t){if(t=t||[],"[object Array]"===Object.prototype.toString.call(t)){for(var s=0;s<i.length;s++){var o=n(i[s]);a[o].references--}for(var h=r(t,e),l=0;l<i.length;l++){var d=n(i[l]);0===a[d].references&&(a[d].updater(),a.splice(d,1))}i=h}}}}}]);