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

/* Simple routing system */
define([], function () {
    "use strict";
    var routes = [
        {hash: '#home', controller: 'HomeView'},
        {hash: '#dashboard', controller: 'DashboardView'},
        {hash: '#project', controller: 'NavbarView'}
    ];
    var currentHash = '';
    var currentState = null;

    function start() {
        setInterval(hashCheck, 200);
    }

    function hashCheck() {
        if (_.isUndefined(window.history.state)) {
            redirectToRoot();
            return;
        }

        if (window.location.hash !== currentHash ||
            !_.isEqual(window.history.state, currentState)) {
            currentHash = window.location.hash;
            for (var i = 0; i < routes.length; i ++) {
                var currentRoute = routes[i];
                if (window.location.hash === currentRoute.hash) {
                    loadController(currentRoute.controller);

                    if (window.history.state === null) {
                        window.history.pushState(currentState, null, this.url);
                    } else {
                        currentState = window.history.state;
                    }
                    break;
                }
            }
        }
    }

    function loadController(controllerName) {
        require(['mvc/' + controllerName], function (controller) {
            controller.start();
        });
    }

    function redirect(url, state) {
        window.history.pushState(state, null, url);
    }

    function redirectToRoot() {
        window.history.pushState(null, null, "#project");
    }

    return {
        start : start,
        redirect : redirect,
        redirectToRoot : redirectToRoot
    };
});
