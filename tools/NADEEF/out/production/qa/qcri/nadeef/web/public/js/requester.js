define(['router', 'state', 'ruleminer', 'blockUI'], function (Router, State) {
    "use strict";
    var cache = {};

    function get(call) {
        if (!('type' in call)) {
            call.type = 'GET';
        }

        if (!('dataType' in call)) {
            call.dataType = 'json';
        }

        var key = 'key' in call ? call.key : call.url + call.type;
        var promise = cache[key];
        if (!promise) {
            promise = $.ajax(call);
            cache[key] = promise;
        } else {
            console.log("Reusing cache " + key);
        }
        return { key: key, promise: promise };
    }

    function request(obj, callbacks) {
        var key = obj.key;
        var promise = obj.promise;
        if (_.isNull(callbacks) || _.isUndefined(callbacks)) {
            $.when(promise).always(function () {
                delete cache[key];
            });
        } else {
            if (_.isUndefined(callbacks.blockUI) || true === callbacks.blockUI) {
                $.blockUI({ baseZ: 2000 });
            }

            if (callbacks.before) {
                callbacks.before();
            }
            $.when(promise).done(function (data) {
                if (callbacks.success) {
                    callbacks.success(data);
                }
            }).fail(function (data) {
                if (callbacks.failure) {
                    callbacks.failure(data);
                }
            }).always(function (data) {
                if (callbacks.always) {
                    callbacks.always(data);
                }

                delete cache[key];
                if (_.isUndefined(callbacks.blockUI) || callbacks.blockUI === true) {
                    $.unblockUI();
                }
            });
        }
        return promise;
    }

    function getProjectName() {
        return State.get('project');
    }

    function getFilter() {
        return State.get('filter');
    }

    function getViolationMetaData(rule, x) {
        return request(
            get({
                url: "/" + getProjectName() + "/violation/metadata",
                data: "rule=" + rule[0]
            }), x);
    }

    function getProgress(x) {
        return request(get({ url : "/progress", type: "GET" }), x);
    }

    function getProject(x) {
        return request(get({ url : "/project"}), x);
    }

    function doDetect(data, x) {
        // inject project name
        data.project = getProjectName();
        return request(get({
            url : "/do/detect",
            type : "POST",
            data: data,
            key: "detect-" + data.project + "-" + data.name
        }), x);
    }

    function deleteViolation(x) {
        return request(
            get({ url : "/" + getProjectName() + "/table/violation", type: "DELETE"}), x);
    }

    function getSource(x) {
        return request(get({ url : '/' + getProjectName() + '/data/source'}), x);
    }

    function getRule(x) {
        return request(get({ url : '/' + getProjectName() + '/data/rule'}), x);
    }

    function deleteRule(data, x) {
        data.project = getProjectName();
        return request(get({
            url : "/" + getProjectName() + "/data/rule",
            type: "DELETE",
            data: data
        }), x);
    }

    function getRuleDetail(ruleName, x) {
        return request(get({url : '/' + getProjectName() + '/data/rule/' + ruleName}), x);
    }

    function getTableSchema(tableName, x) {
        return request(get({url : '/' + getProjectName() + '/table/' + tableName + '/schema'}), x);
    }

    function getOverview(successCallback, failureCallback) {
        $.ajax({
            url : '/' + getProjectName() + '/widget/overview',
            type: 'GET',
            success: successCallback,
            error: failureCallback
        });
    }

    function getAttribute(x) {
        return request(
            get({
                url: '/' + getProjectName() + '/widget/attribute',
                data: 'filter=' + getFilter()
            }), x);
    }

    function getViolationRelation(x) {
        return request(
            get({
                url: '/' + getProjectName() + '/widget/violation_relation',
                data: 'filter=' + getFilter()
            }), x);
    }

    function getRuleDistribution(successCallback, failureCallback) {
        $.ajax({
            url : '/' + getProjectName() + '/widget/rule',
            type: 'GET',
            success: successCallback,
            error: failureCallback
        });
    }

    function getTupleRank(successCallback, failureCallback) {
        $.ajax({
            url : '/' + getProjectName() + '/widget/top/100',
            type: 'GET',
            success: successCallback,
            error: failureCallback
        });
    }

    function doVerify(data, x) {
        // inject project name
        data.project = getProjectName();
        return request(
            get({ url : "/do/verify", type: "POST", data : data }), x);
    }

    function doRepair(data, successCallback, failureCallback) {
        // inject project name
        data.project = getProjectName();
        $.ajax({
            url : '/do/repair',
            type: 'POST',
            dataType: 'json',
            data: data,
            success: successCallback,
            error: failureCallback
        });
    }

    function doGenerate(data, x) {
        // inject project name
        data.project = getProjectName();
        return request(get({ url : "/do/generate", type: "POST", data : data }), x);
    }

    function createRule(data, x) {
        // inject project name
        data.project = getProjectName();
        return request(
            get({ url : "/" + getProjectName() + "/data/rule", type: "POST", data : data }), x);
    }

    function createProject(projectName, x) {
        return request(
            get({ url: "/project", type: "POST", data: { project : projectName }}), x);
    }

    function getNotebooks(x) {
        return request(get({ url: "/analytic/notebooks", type: "GET"}), x);
    }

    function createNotebook(projectName, x) {
        return request(get({ url: "analytic/" + projectName, type: "POST"}), x);
    }

    return {
        doDetect : doDetect,
        doVerify : doVerify,
        doRepair : doRepair,
        doGenerate : doGenerate,
        deleteViolation : deleteViolation,
        deleteRule: deleteRule,
        createProject: createProject,
        createRule: createRule,
        getProgress : getProgress,
        getProject: getProject,
        getSource: getSource,
        getRule: getRule,
        getRuleDetail: getRuleDetail,
        getTableSchema: getTableSchema,
        getOverview: getOverview,
        getAttribute: getAttribute,
        getViolationRelation: getViolationRelation,
        getRuleDistribution: getRuleDistribution,
        getTupleRank: getTupleRank,
        getViolationMetaData: getViolationMetaData,
        getNotebooks: getNotebooks,
        createNotebook: createNotebook
    };
});