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
    var keyList = [ "project", "job", "source", "currentSource", "rule", "currentRule", "filter"];
    var changeHandler = {};

    function set(key, value) {
        if (_.indexOf(keyList, key) === -1) {
            console.log("Key is not found.");
            return;
        }

        if (_.isObject(value) && !_.isArray(value)) {
            window.localStorage.setItem(key, JSON.stringify(value));
        } else {
            set(key, { _: value});
        }

        if (key in changeHandler) {
            var fs = changeHandler[key];
            _.each(fs, function (f) { f(); });
        }
    }

    function get(key) {
        if (_.indexOf(keyList, key) === -1) {
            console.log("Key is not found.");
            return;
        }

        var item = window.localStorage.getItem(key);
        if (_.isUndefined(item) || _.isNull(item)) {
            return null;
        }

        var obj = JSON.parse(window.localStorage.getItem(key));
        if ('_' in obj) {
            return obj._;
        }

        return obj;
    }

    function containsKey(key) {
        return _.indexOf(keyList, key) > -1;
    }

    function subscribe(key, f) {
        if (!containsKey(key)) {
            console.log("Key is not found.");
            return;
        }

        if (key in changeHandler) {
            var subscriber = changeHandler[key];
            subscriber.push(f);
        } else {
            changeHandler[key] = [f];
        }
    }

    function clear(key) {
        if (!containsKey(key)) {
            console.log("Key is not found.");
            return;
        }
        delete window.localStorage[key];
    }

    function init() {
        set('job', []);
        set('rule', []);
        set('source', []);
    }

    return {
        init : init,
        get : get,
        set : set,
        clear : clear,
        subscribe : subscribe
    };
});
