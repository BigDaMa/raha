/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

define([], function () {
    "use strict";
    var instance = null;
    var pingId = null;
    var MAX_PING = 20;
    var hostname = getHostname();
    var baseUrl = hostname + "/DenialConstraintDemoUI/index.action?";

    // TODO: move to the backend.
    function getHostname() {
        var windowHostname = window.location.hostname;
        if (windowHostname.indexOf("localhost") > -1) {
            return "http://localhost:8080";
        }

        if (windowHostname.indexOf("qcridemos.org") > -1) {
            return "http://ruleminer.da.qcridemos.org";
        }

        return "http://" + window.location.hostname + ":8080";
    }

    // TODO: move to the backend.
    function isAvailable() {
        return true;
    }

    function isWindowOpened() {
        return instance != null && instance.closed === false;
    }

    function start(project, table) {
        // TODO: to improve
        var url = baseUrl + 'datasource=' + project + '&datasetname=' + table.substr(3);
        if (isAvailable() && !isWindowOpened()) {
            instance = window.open(url, "Rule Miner");
            $.blockUI({ message: "<h4>Starting Rule Miner</h4>"});
            var i = 0;
            var pingFunc = function () {
                i ++;
                if (i >= MAX_PING) {
                    if (pingId) {
                        clearInterval(pingId);
                    }

                    $.blockUI({message: "<h4>Rule Miner failed to start.</h4>"});
                    setTimeout(function () {$.unblockUI(); }, 1500);
                    console.log("Starting rule miner failed.");
                } else {
                    ping();
                }
            };

            window.addEventListener("message", function (event) {
                console.log('received: ' + event.data);
                if (event.origin.indexOf(hostname) > -1) {
                    if (event.data.indexOf("pong") > -1) {
                        if (pingId) {
                            clearInterval(pingId);
                        }
                        $.unblockUI();
                    }

                    if (event.data.indexOf("salam") > -1) {
                        instance = null;
                    }
                }
            }, false);

            pingId = setInterval(pingFunc, 200);
            return true;
        }
        return false;
    }

    function close() {
        if (isWindowOpened()) {
            instance.close();
        }
        instance = null;
    }

    function ping() {
        if (isAvailable() && isWindowOpened()) {
            instance.postMessage("ping", "*");
        }
    }

    return {
        isAvailable : isAvailable,
        isWindowOpened: isWindowOpened,
        ping: ping,
        close: close,
        start: start
    };
});