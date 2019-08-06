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
define(["requester", "state"], function (Requester, State) {
    "use strict";
    var sourceInstance = null;
    var auditInstance = null;
    var violationInstance1 = null;
    var violationInstance2 = null;
    var sourceUrl = null;
    var auditUrl = null;

    function loadSourceTable(domId, tablename) {
        $("#source-table-info").addClass('hide');
        $('#source-table').removeClass('hide');
        var url = "/" + State.get('project') + "/table/" + tablename;
        var columnDefs = [];

        if (sourceUrl === url) {
            sourceInstance.ajax.reload();
        } else {
            Requester.getTableSchema(tablename, {
                success: function (data) {
                    var schema = data.schema;
                    var columns = [];
                    var dom = $('#' + domId);
                    if ($.fn.dataTable.isDataTable('#' + domId)) {
                        dom.DataTable().destroy();
                    }
                    dom.empty().append('<thead><tr></tr></thead>');
                    for (var i = 0; i < schema.length; i++) {
                        columns.push({ sTitle: schema[i] });
                        columnDefs.push({ "defaultContent" : "" });
                    }

                    var search = State.get("filter") ? State.get("filter") : "";
                    sourceInstance = dom.DataTable({
                        "scrollX": true,
                        "ordering": false,
                        "processing": true,
                        "serverSide": true,
                        "ajax": { "url": url, "dataSrc": 'data' },
                        "search": { "search": search },
                        "destroy": true,
                        "columns": columns,
                        "ajaxDataProp": 'data',
                        "columnDefs" : columnDefs
                    });
                }
            });
            sourceUrl = url;
        }
    }

    function loadAuditTable(domId, tablename) {
        var url = "/" + State.get('project') + "/table/" + tablename;

        if (auditUrl === url) {
            auditInstance.ajax.reload();
        } else {
            Requester.getTableSchema(tablename, {
                success: function (data) {
                    var schema = data.schema;
                    var columns = [];
                    var columnDefs = [];
                    var dom = $('#' + domId);
                    if ($.fn.dataTable.isDataTable('#' + domId)) {
                        dom.DataTable().destroy();
                    }
                    dom.empty().append('<thead><tr></tr></thead>');
                    for (var i = 0; i < schema.length; i++) {
                        columns.push({ sTitle: schema[i] });
                        columnDefs.push({ "defaultContent" : "" });
                    }
                    var project = State.get('project');
                    var search = State.get("filter") ? State.get("filter") : "";

                    auditInstance = dom.DataTable({
                        "scrollX": true,
                        "ordering": false,
                        "processing": true,
                        "serverSide": true,
                        "ajax": { "url": url, "dataSrc": 'data' },
                        "search": { "search": search },
                        "destroy": true,
                        "columns": columns,
                        "columnDefs" : columnDefs,
                        "ajaxDataProp": 'data'
                    });
                }
            });
            auditUrl = tablename;
        }
    }

    function load(table) {
        if (_.isNull(table) || _.isNaN(table)) {
            console.log("Table name is null or empty.");
            return;
        }

        switch (table.domId) {
            case "source-table":
                loadSourceTable(table.domId, table.table, table.rule);
                break;
            case "violation-table":
                loadViolationTable(table.domId, table.table, table.rule);
                break;
            case "audit-table":
                loadAuditTable(table.domId, table.table, table.rule);
                break;
        }
    }

    function createViolationTable(domId, tableName, rule, isFirst, length) {
        var schema; // evil closure
        Requester.getTableSchema(tableName, {
            success: function (data) {
                var dom = $('#' + domId);
                var columns = [];
                var columnDefs = [];
                var instance;
                var project = State.get('project');
                schema = data.schema;

                if ($.fn.dataTable.isDataTable('#' + domId)) {
                    dom.DataTable().destroy();
                }

                dom.empty().append('<thead><tr></tr></thead>');
                for (var i = 0; i < schema.length; i++) {
                    columns.push({ sTitle: schema[i] });
                    columnDefs.push({ "defaultContent" : "" });
                }

                var ruleQuery = "";
                for (i = 0; i < rule.length; i++) {
                    ruleQuery += "rule=" + rule[i];
                }

                var url = "/" + project + "/violation/" + tableName + "?" + ruleQuery;
                var search = State.get("filter") ? State.get("filter") : "";

                instance = dom.DataTable({
                    "createdRow": function (row, data) {
                        var vid = data[data.length - 2];
                        var attrToken = data[data.length - 1];
                        var rainbow = ["#dff0d8", "#fcf8e3", "#d9edf7", "#f2dede"];
                        // custom parsing
                        var cells = $('td', row);
                        cells.attr('data', vid);
                        for (var i = 0, pre = 0; i < attrToken.length; i++) {
                            if ('{' === attrToken[i] ||
                                '}' === attrToken[i] ||
                                ',' === attrToken[i]) {
                                var attr = attrToken.substring(pre, i);
                                if (!_.isEmpty(attr)) {
                                    var cell = cells.eq(_.indexOf(schema, attr));
                                    cell.css("background-color", rainbow[vid % rainbow.length])
                                        .attr("data-toggle", "tooltip")
                                        .attr("font-weight", "bolder")
                                        .attr("title", "vid: " + vid);
                                }
                                pre = i + 1;
                            }
                        }

                        // create event for click filter
                        cells.on("click", function () {
                            var vid = this.getAttribute("data");
                            if (violationInstance1 != null) {
                                violationInstance1.search(":=" + vid).draw();
                            }

                            if (violationInstance2 != null) {
                                violationInstance2.search(":=" + vid).draw();
                            }
                        });

                        // truncate large text
                        for (i = 0; i < data.length; i ++) {
                            if (!_.isEmpty(data[i]) && data[i].length > 300) {
                                var label = '... <span class="label label-info" ' +
                                    'data-toggle="tooltip" ' +
                                    'title="' + data[i] + '">More</span>';
                                cells.eq(i).html(data[i].substring(0, 300) + label);
                            }
                        }
                    },
                    "initComplete": function () {
                        $(document).on('keyup', keyPressHandler);
                    },
                    "scrollX": true,
                    "ordering": false,
                    "processing": true,
                    "serverSide": true,
                    "ajax": { "url": url, "dataSrc": 'data' },
                    "destroy": true,
                    "columns": columns,
                    "columnDefs" : columnDefs,
                    "search": { "search" : search },
                    "ajaxDataProp": 'data',
                    "lengthMenu": _.isUndefined(length) ? [10, 25, 50] : length
                });

                instance.tableName = tableName;
                if (isFirst) {
                    violationInstance1 = instance;
                } else {
                    violationInstance2 = instance;
                }
            }
        });
    }

    function keyPressHandler(e) {
        if (e.which === 27) {
            if (violationInstance1 != null) {
                violationInstance1.search("").draw();
            }

            if (violationInstance2 != null) {
                violationInstance2.search("").draw();
            }
        }
    }

    // render of violation table
    function loadViolationTable(domId, tablename, rule) {
        $("#violation-table-info").addClass('hide');
        $('#violation-table-div').removeClass('hide');

        Requester.getViolationMetaData(rule, {
            success: function (data) {
                var tables =
                    data.data.length < 1 ?
                        [[0, State.get("currentSource")]] : data.data;
                if (tables.length > 1) {
                    createViolationTable(
                        domId,
                        tables[0][1],
                        rule,
                        true,
                        [3, 10, 25, 50]
                    );
                    createViolationTable(
                        "violation-table-extra",
                        tables[1][1],
                        rule,
                        false,
                        [3, 10, 25, 50]
                    );
                    $("#violation-table-extra_wrapper").removeClass("hide");
                    $("#violation-table-extra").removeClass("hide");
                } else {
                    if ($.fn.dataTable.isDataTable('#violation-table-extra')) {
                        $("#violation-table-extra").DataTable().destroy();
                        $("#violation-table-extra").addClass('hide');
                        $("#violation-table-extra_wrapper").addClass('hide');
                    }
                    createViolationTable(domId, tables[0][1], rule);
                }
            }
        });
    }

    function filter(e) {
        if (violationInstance1) {
            violationInstance1.search(e).draw();
        }

        if (violationInstance2) {
            violationInstance2.search(e).draw();
        }
    }

    function filterByCluster(e) {
        if (!_.isArray(e)) {
            console.log("Input is not an array.");
            return;
        }

        var query = {},
            tableName,
            ids,
            str;

        for (var i = 0; i < e.length; i ++) {
            tableName = e[i].tableName;
            if (tableName in query) {
                ids = query[tableName];
            } else {
                query[tableName] = [];
                ids = query[tableName];
            }
            ids.push(e[i].tid);
        }

        var tables = Object.keys(query);
        for (var j = 0; j < tables.length; j ++) {
            tableName = tables[j];
            str = '?=';
            ids = query[tableName];
            for (var k = 0; k < ids.length; k ++) {
                if (k !== 0) {
                    str += ',';
                }
                str += ids[k];
            }

            if (violationInstance1 && violationInstance1.tableName === tableName) {
                violationInstance1.search(str).draw();
            } else if (violationInstance2 && violationInstance2.tableName === tableName) {
                violationInstance2.search(str).draw();
            } else {
                console.log("Table name " + tableName + " is not found.");
            }
        }
    }

    return {
        load : load,
        filter : filter,
        filterByCluster : filterByCluster
    };
});

