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

define(["requester"], function (Requester) {
    "use strict";
    var instance;

    // TODO: move to backend
    function getHostName() {
        var windowHostname = window.location.hostname;
        var urlBase = null;
        if (windowHostname.indexOf("localhost") > -1) {
            urlBase = "http://localhost:8888";
        } else if (windowHostname.indexOf("qcridemos.org") > -1) {
            urlBase = "http://notebook.da.qcridemos.org:1067";
        } else {
            urlBase = "http://" + window.location.hostname + ":8888";
        }
        return urlBase;
    }

    // TODO: move to the backend.
    function isAvailable() {
        return true;
    }

    function isWindowOpened() {
        return instance != null && instance.closed === false;
    }

    function start(project) {
        var url = getHostName();
        var projectName = project + ".ipynb";
        var projectUrl = url + "/notebooks/" + projectName;
        if (isAvailable() && !isWindowOpened()) {
            var promise = Requester.getNotebooks();
            $.when(promise).done(function (json) {
                var existsProject = _.find(json, function (x) {
                    return x.name === projectName;
                });

                if (existsProject) {
                    instance = window.open(projectUrl, "NADEEF Analytic");
                } else {
                    Requester.createNotebook(projectName, {
                        success: function () {
                            instance = window.open(projectUrl, "NADEEF Analytic");
                        },
                        failure: function () {
                            console.log("Creating new notebook failed.");
                            $.blockUI({
                                message: '<h3>Creating NADEEF Notebook.</h3>',
                                timeout: 1500
                            });
                        }
                    });
                }
            }).fail(function () {
                $.blockUI({
                    message: '<h3>NADEEF Notebook failed to start.</h3>',
                    timeout: 1500
                });
            });
        } else {
            $.blockUI({
                message: '<h3>NADEEF Notebook is already opened.</h3>',
                timeout: 1000
            });
        }
    }

    function close() {
        if (isWindowOpened()) {
            instance.close();
        }
        instance = null;
    }

    return {
        isAvailable : isAvailable,
        isWindowOpened: isWindowOpened,
        close: close,
        start: start
    };
});