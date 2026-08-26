def decode_launch_messages(packages, locale):
    def safe_path(s):
        if not isinstance(s, str) or not s or s[0] == "\\" or ":" in s or "\x00" in s:
            return False
        parts = s.split("\\")
        return all(p not in ("", ".", "..") for p in parts)

    def normalize_name(s):
        return " ".join(s.strip().split())

    fallback = []
    seen_locales = set()
    cur_locale = locale
    while cur_locale:
        cf = cur_locale.casefold()
        if cf not in seen_locales:
            seen_locales.add(cf)
            fallback.append(cf)
        if "-" not in cur_locale:
            break
        cur_locale = cur_locale.rsplit("-", 1)[0]
    if "*" not in seen_locales:
        fallback.append("*")

    package_by_name = {}
    selected_resources = []
    asset_exact = []
    asset_scaled = []

    for pi, package in enumerate(packages):
        package_by_name[package["name"].casefold()] = pi

        tables = {}
        for tag, values in package["resources"].items():
            tables[tag.casefold()] = values

        selected = {}
        for tag in fallback:
            table = tables.get(tag)
            if table is not None:
                for key, value in table.items():
                    if key not in selected:
                        selected[key] = value
        selected_resources.append(selected)

        exact = {}
        scaled = {}
        for asset in package["assets"]:
            if not safe_path(asset):
                continue

            acf = asset.casefold()
            if acf not in exact:
                exact[acf] = asset

            pos = asset.rfind("\\")
            directory = asset[:pos] if pos >= 0 else ""
            filename = asset[pos + 1:]
            dot = filename.rfind(".")
            if dot <= 0:
                continue

            ext = filename[dot:].casefold()
            base = filename[:dot]
            marker = base.rfind(".scale-")
            if marker < 0:
                continue

            stem = base[:marker]
            digits = base[marker + 7:]
            if not digits or (len(digits) > 1 and digits[0] == "0"):
                continue
            if any(c < "0" or c > "9" for c in digits):
                continue

            n = int(digits)
            if n < 1 or n > 1000:
                continue

            key = (directory.casefold(), stem.casefold(), ext)
            old = scaled.get(key)
            if old is None or n > old[0]:
                scaled[key] = (n, asset)

        asset_exact.append(exact)
        asset_scaled.append(scaled)

    resource_state = {}
    resource_value = {}
    missing = object()

    def resolve_resource(start_pi, start_key):
        start = (start_pi, start_key)
        if resource_state.get(start, 0) == 2:
            return resource_value[start]

        stack = []
        cur = start

        while True:
            state = resource_state.get(cur, 0)

            if state == 2:
                value = resource_value[cur]
                break

            if state == 1:
                for node in stack:
                    resource_state[node] = 2
                    resource_value[node] = None
                return None

            pi, key = cur
            resource_state[cur] = 1
            stack.append(cur)

            raw = selected_resources[pi].get(key, missing)
            if raw is missing:
                value = None
                break

            if not raw.startswith("@"):
                value = raw
                break

            body = raw[1:]
            if body.count("/") > 1:
                value = None
                break

            if "/" in body:
                package_name, target_key = body.split("/", 1)
                if not package_name or not target_key:
                    value = None
                    break
                target_pi = package_by_name.get(package_name.casefold())
                if target_pi is None:
                    value = None
                    break
                cur = (target_pi, target_key)
            else:
                if not body:
                    value = None
                    break
                cur = (pi, body)

        while stack:
            node = stack.pop()
            resource_state[node] = 2
            resource_value[node] = value

        return value

    def resolve_field(pi, value):
        if value is None:
            return None
        if not value.startswith("@"):
            return value

        body = value[1:]
        if body.count("/") > 1:
            return None

        if "/" in body:
            package_name, key = body.split("/", 1)
            if not package_name or not key:
                return None
            target_pi = package_by_name.get(package_name.casefold())
            if target_pi is None:
                return None
            return resolve_resource(target_pi, key)

        if not body:
            return None
        return resolve_resource(pi, body)

    def find_icon(pi, values):
        for preferred_source in values:
            preferred = resolve_field(pi, preferred_source)
            if preferred is None or not safe_path(preferred):
                continue

            low = preferred.casefold()
            if not (low.endswith(".png") or low.endswith(".ico")):
                continue

            exact = asset_exact[pi].get(low)
            if exact is not None:
                return exact

            pos = preferred.rfind("\\")
            directory = preferred[:pos] if pos >= 0 else ""
            filename = preferred[pos + 1:]
            ext = filename[-4:]
            stem = filename[:-4]

            scaled = asset_scaled[pi].get(
                (directory.casefold(), stem.casefold(), ext.casefold())
            )
            if scaled is not None:
                return scaled[1]

        return None

    result = []
    seen_identities = set()

    for pi, package in enumerate(packages):
        if package["store"] is not True or package["kind"] != "application":
            continue

        for app in package["apps"]:
            if app["maintenance"]:
                continue

            app_id = app["id"]
            if not app_id:
                continue

            valid_id = True
            for ch in app_id:
                if not (
                    "A" <= ch <= "Z"
                    or "a" <= ch <= "z"
                    or "0" <= ch <= "9"
                    or ch in "._-"
                ):
                    valid_id = False
                    break
            if not valid_id:
                continue

            executable = resolve_field(pi, app["executable"])
            if not safe_path(executable) or not executable.casefold().endswith(".exe"):
                continue

            name = None
            for candidate in (
                app["display"],
                app["short_name"],
                package["display"],
            ):
                resolved = resolve_field(pi, candidate)
                if resolved is None:
                    continue
                normalized = normalize_name(resolved)
                if normalized:
                    name = normalized
                    break

            if name is None:
                name = app_id

            working = resolve_field(pi, app["working"])
            if not safe_path(working):
                pos = executable.rfind("\\")
                working = executable[:pos] if pos >= 0 else "."

            icon = find_icon(
                pi,
                (app["icon"], app["logo"], package["icon"])
            )

            identity = package["name"] + "!" + app_id
            identity_cf = identity.casefold()
            if identity_cf in seen_identities:
                continue
            seen_identities.add(identity_cf)

            result.append({
                "identity": identity,
                "name": name,
                "target": executable,
                "arguments": "" if app["arguments"] is None else app["arguments"],
                "working": working,
                "icon": icon
            })

    return result