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

require.config({
    baseUri: '.',
    waitSeconds: 0,
    paths: {
        "text" : "lib/text",
        "jquery" : "lib/jquery-1.10.2.min",
        "underscore" : "lib/underscore.min",
        "bootstrap" : "lib/bootstrap.min",
        "datatables" : "lib/jquery.dataTables.min", // has to be this name
        "d3" : "lib/d3.min",
        "nvd3" : "lib/nv.d3.min",
        "ace" : "lib/ace-min/ace",
        "jquery.filedrop" : "lib/jquery.filedrop",
        "blockUI" : "lib/jquery.blockUI"
    },

    shim: {
        'underscore' : {
            exports: '_'
        },
        'ace' : {
            deps : ['jquery'],
            exports: 'ace'
        },

        'd3'        : ['jquery'],
        'nvd3'      : ['d3'],
        'blockUI'   : ['jquery'],
        'bootstrap' : ['jquery'],
        "datatables-bootstrap" : ['jquery', 'datatables']
    }
});

// main start

require(
    ['router', 'state', 'bootstrap', 'underscore', 'datatables'],
    function (Router, State) {
        "use strict";
        // Check for datatable and bootstrap loading status
        //
        // console.log(jQuery.fn.DataTable);
        // console.log((typeof $().modal == 'function'));
        //
        State.init();
        Router.start();
        Router.redirect('#project');
    }
);
